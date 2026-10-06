# Test Specification — LOG-C2-005

**Template ID:** LOG-C2-005
**Template Name:** LogisticsRegulationQAAgent
**Category:** Cat 2 (nested RAG)

This document defines the test cases for the template (state, nodes,
inner/outer graphs, manifest, server). The test code lives in `tests/unit/` +
`tests/proof_of_boundary/`; this spec is the contract those tests implement.

## 1. Scope & Invocation Conventions

- Per-node unit tests for the 5 inner domain nodes + the 2 outer gate nodes.
- Manifest/config consistency (`config/agent.yaml` + `config/config.yaml` <->
  code) and seeded-KB integrity.
- The caller-parameter contract (fail-closed validation of `input_context` /
  JSON-envelope fields, including the non-finite numeric matrix).
- Retrieval quality (golden queries over `config/kb/logistics_regulation_kb.json`,
  one query per seeded KB entry) including the hybrid exact regulation-code
  match bonus.
- Inner-graph (`DomainWorkflowGraph`) and outer-graph
  (`LogisticsRegulationQAAgent`) composition / integration.
- Proof-of-Boundary (PoB): import isolation, State msgpack safety, invoke
  order (PB-6), end-to-end `/invoke` behaviour, HITL propagation (PB-7,
  conditional), server boot.

**Trust-gate invocation canon.** Every per-node test invokes
the node via `node(state)` — through `BaseNode.__call__`, which runs the
trust gate -> input mask -> `execute()` -> credential scan — never a bare
`node.execute(state)`. The state builder sets `caller_trust_level` to
`TrustLevel.VERIFIED_EXTERNAL.value` for the two outer gate slots
(PreProcessNode / PostProcessNode — the manifest's declared caller level) and
`TrustLevel.ANONYMOUS.value` for the five inner domain nodes.

**No config parameter.** Nodes do not accept a `config` parameter —
`execute(self, state)` is the only signature; a call passing a 2nd argument
raises `TypeError`. Every config knob (`top_k`, `score_threshold`, `kb_path`,
`hybrid_search`) is exercised by SEEDING the state field `retrieval_config`
(the same field `DomainWorkflowGraph._extra_initial_state()` republishes from
`config/config.yaml` via `LogisticsRegulationSearchGraphNode._parent_config()`)
and still invoking through `node(state)`. There is no "passed config beats
state" precedence tier in this template — only state `retrieval_config` >
module defaults.

**Input-mask expectations.** The framework input gate masks
`user_input`/`validated_input`/`llm_response` (e-mail, phone/SSN/CC digit
groups, Title-Case name bigrams) to `[MASKED]` before `execute()` runs.
Positive-path payloads are therefore lowercase, PII-free regulatory phrasing;
intentional-PII tests assert the raw identifier is gone and `[MASKED]` is
present. `PreProcessNode` has no node-level identifier screen of its own (no
`[REDACTED]` path) — only the framework input gate applies. Domain fields
(`grounded_answer`, `formatted_answer`, `retrieved_documents`, …) are not
input-mask targets.

**Audit muting.** `shared.*` is never sys.modules-stubbed (the framework
imports `shared.security` at load time). The domain audit emitter is muted via
an autouse fixture patching `src.nodes.<mod>.emit_trace_event`; the audit
assertion test re-patches the same attribute with a spy and asserts on
`call.args[1]` (the event payload).

## 2. Unit Test Cases

### 2.1 PreProcessNode (outer pre_process slot; input gate) — `test_pre_process_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| PRE-01 | Valid query | lowercase regulatory question | `status=SUCCESS`, `validated_input` set, `enriched_context` carries channel/source |
| PRE-02 | Empty input | `""` / whitespace | `status=ERROR`, `error_log` non-empty, no `validated_input` |
| PRE-03 | Missing / non-string input | `user_input` absent; dict payload | `status=ERROR` |
| PRE-04 | Identifier screen | e-mail / 4-4-4 digit groups -> framework input mask | raw identifier absent from `validated_input`; `[MASKED]` present |
| PRE-08 | Domain audit | valid query | `pre_process_complete` emitted; payload (`call.args[1]`) carries `input_chars` |

### 2.2 InputValidateNode (inner node 1) — `test_input_validate_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| VAL-01 | Plain text | free-text query | whole string becomes `search_query`; filters `{category: None, top_k: None}` |
| VAL-02 | Whitespace | ragged spacing/newlines | collapsed to single spaces |
| VAL-03 | JSON envelope | `{"query","category","top_k"}` | all three parsed; `question` alias accepted; category lower-cased/stripped |
| VAL-04 | Malformed JSON | `{`-prefixed non-JSON | treated as plain-text query + parse note |
| VAL-05 | top_k fail-closed (envelope) | parametrized invalid matrix: `"NaN"` / `"Infinity"` / `"-Infinity"` / raw `NaN` / raw `±Infinity` / 0 / -5 / 21 / 99 / 3.5 / `true` / `false` / `"many"` / list / dict | `status=error`; error names `top_k` (accepted range only — value never echoed); no `search_query`/`query_filters` emitted |
| VAL-06 | top_k fail-closed (input_context) | same matrix via `input_context` | identical rejection |
| VAL-07 | Integral float | `top_k=4.0` | accepted as 4 |
| — | input_context params | `{"category", "top_k"}` via `input_context` | used; wins over the envelope when both supply a field; unknown keys ignored |
| — | category fail-closed | non-identifier / oversize / non-string category | `status=error`; error names `category`; value never echoed; blank category = absent |
| — | multi-field rejection | invalid category AND top_k | single rejection naming both fields |
| VAL-08 | Oversize query | > 2000 chars | truncated to 2000 + note |
| VAL-09 | Empty request | `""` | `search_query=""` + "empty request" note (non-fatal) |
| — | JSON-string state | any | `query_filters` is a JSON string, never a bare dict |

### 2.3 RetrieveNode (inner node 2) — `test_retrieve_node.py`

Hybrid retrieval: the usual field-weighted keyword score (title 1.0 > tags 0.8
> content 0.5) PLUS a `+0.5` exact regulation-code match bonus when a query
token matches an entry's declared `codes` list (UN dangerous-goods numbers,
HS tariff headings, IATA/IMDG designators) and `hybrid_search` is enabled.

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RET-01 | Happy path | dangerous-goods (lithium battery / IMDG) query | top-1 candidate is `kb-001` |
| RET-02 | Ordering | same query | scores strictly sorted desc; all > 0 |
| RET-03 | Entry shape | any hit | keys `{id,title,category,source,score,excerpt}`; excerpt <= 400 chars |
| RET-04 | Category filter | `query_filters.category="customs"` | only customs entries; top-1 `kb-002` |
| RET-05 | Empty query | `""` | no candidates |
| RET-06 | State kb_path override | bad path in state `retrieval_config` | `[]` + "not readable" note |
| RET-07 | Malformed state top_k | non-numeric `top_k` in state `retrieval_config` | falls back to default `top_k`, no crash |
| RET-08 | Notes accumulation | prior `intake_notes` | appended, never clobbered |
| RET-09 | Hybrid code-match bonus | query = bare code (`"un3480"`) | `hybrid_search=True` (default): `kb-001` hit via bonus alone (score 0.5); `hybrid_search=False` (state override): no candidates |

### 2.4 RerankFilterNode (inner node 3) — `test_rerank_filter_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RRF-01 | Relevance floor | scores 0.9 / 0.1 | 0.1 dropped (default 0.25 floor) |
| RRF-02 | State score_threshold override | `retrieval_config.score_threshold=0.5` | 0.3 dropped |
| RRF-03 | State top_k override | `retrieval_config.top_k=1` | one survivor, highest score |
| RRF-04 | Category boost | matching category | +0.1, re-ranked ahead |
| RRF-05 | Boost cap | 0.95 + boost | capped at 1.0 |
| RRF-06 | Caller top_k | stricter (1) wins; looser (10) does not widen | enforced |
| RRF-07 | Garbage entries | non-dict / uncoercible score | skipped / coerced to 0.0 and dropped |
| RRF-08 | Tie-break | equal scores | deterministic id-ascending order |

### 2.5 GenerateAnswerNode (inner node 4) — `test_generate_answer_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| GEN-01 | Citation markers | 2 ranked passages | `[1]`/`[2]` markers with titles |
| GEN-02 | Lead sentence | query present | query quoted in the lead |
| GEN-03 | Citations list | ranked passages | refs 1..n mirror ranked order; id/title/source carried |
| GEN-04 | Groundedness | single passage | answer body traces to ranked passages only |
| GEN-05 | No coverage | empty/missing `ranked_documents` | escalation answer; `citations=[]` |

### 2.6 OutputFormatNode (inner node 5, terminal) — `test_output_format_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| FMT-01 | Full compose | body + citations | header + body + `## Sources` rows + advisory disclaimer; `status=SUCCESS` |
| FMT-02 | Blank source | citation without source | no `()` suffix |
| FMT-03 | Disclaimer | every input | disclaimer rides with every answer |
| FMT-04 | No citations | empty list | explicit "- none (…)" sources line |
| FMT-05 | Missing body | no `grounded_answer` | fallback text; `status=SUCCESS` |

### 2.7 PostProcessNode (outer post_process slot; output gate) — `test_post_process_node.py`

| ID | Case | Input (`result`) | Expected |
|----|------|------------------|----------|
| POST-01 | Clean output | normal KB answer | `formatted_output=result`, `status=SUCCESS` |
| POST-02 | Empty result | `""` | forwarded as-is, `status=SUCCESS` (non-fatal) |
| POST-03..06 | Credential leak | `sk-` API key / `password=` assignment / JWT (built at runtime) / Bearer token | `formatted_output` + `result` replaced with the sanitised stub, `status=ERROR`, raw secret absent from both |
| — | Every representation | violating `result` alongside `regulation_answer` + `citations` | `regulation_answer` replaced with the stub, `citations` cleared — the blocked content survives under NO field name; clean runs return only changed keys |

### 2.8 Manifest / config consistency — `test_config_manifest.py`

| ID | Case | Expected |
|----|------|----------|
| CFG-01 | Template id | manifest root `id` = `LOG-C2-005`; `namespace` = `log`; `enabled` |
| CFG-02 | Class-name contract | manifest `class` = `src.graph.graph.LogisticsRegulationQAAgent` — module AND class match the graph.py agent; `name` matches the agent's `name` property |
| CFG-03 | Classification | Cat 2 / LOG / RAGAgent / `generation_mode=deterministic` |
| — | Requires blocks | `requires.secrets` = `[]` and `requires.extras` = `[]` (no secret use, no LLM client anywhere in `src/`) |
| CFG-04 | Trust level | manifest `VERIFIED_EXTERNAL` == PreProcessNode & PostProcessNode `required_trust_level` |
| CFG-05 | max_retry | `config/config.yaml` int, `0 <= v < 10` (framework ceiling); hitl not enabled (PB-7 waiver contract) |
| CFG-06 | Retrieval block | `config/config.yaml` `top_k`/`score_threshold`/`hybrid_search` mirror node module defaults; `kb_path` exists |
| CFG-07 | `_parent_config()` | forwards the `config/config.yaml` retrieval + llm blocks; never `{}`; `max_retry` travels unchanged and `timeout_s` travels as `timeout_seconds` |
| — | KB integrity | JSON list >= 5 entries; unique ids; required keys per entry (`codes` optional); coded entries carry lowercase code strings |

### 2.9 Retrieval quality (golden queries) — `test_retrieval_quality.py`

| ID | Case | Expected |
|----|------|----------|
| QUAL-01 | 10 golden domain queries (one per seeded KB entry) | expected KB entry is top-1 (kb-001..kb-010) |
| QUAL-02 | Relevance floor | every survivor >= 0.25 |
| QUAL-03 | Citation integrity | every survivor id exists in the seeded KB |
| QUAL-04 | Precision | dangerous-goods query keeps ONLY `kb-001` |
| QUAL-05 | Category filter | customs filter -> only customs entries, top-1 `kb-005` |
| QUAL-06 | No coverage | out-of-domain query -> zero survivors |
| QUAL-07 | Escalation answer | no-coverage -> explicit escalation text, no citations |

## 3. Integration / Composition

### 3.1 Inner graph — `test_domain_workflow_graph.py`

| ID | Case | Expected |
|----|------|----------|
| INT-01 | Composition | inherits `BaseGraph`; registers exactly the 5 domain nodes; no initialize/finalize |
| INT-02 | Config forwarding | `_extra_initial_state()` republishes the retrieval block as the JSON-string `retrieval_config` AND seeds `input_context` from the context bridge |
| INT-03 | Output shape | `get_output()` emits `formatted_answer`/`citations`/`status`/`error_log`/… (the merge contract); `route()` -> END on error, `retrieve` otherwise |
| INT-04 | Inner e2e | full inner `invoke()` -> SUCCESS; formatted answer + disclaimer + kb-001 citation; inner `node_history` = the 5 domain nodes in linear order |
| — | Fail-closed e2e | invalid `top_k` envelope -> `status=error`; no `formatted_answer`; error names the field; `node_history` stops at `InputValidateNode` |

### 3.2 Outer graph + e2e — `test_graph_composition.py`

| ID | Case | Expected |
|----|------|----------|
| INT-05 | Outer composition | inherits `AgentBaseGraph` directly; `Graph` alias; `add_edges()` NOT overridden |
| INT-06 | Backbone slots | compile() fills all 5; pre/main/post are PreProcessNode / LogisticsRegulationSearchGraphNode / PostProcessNode |
| INT-07 | `get_subgraph()` | returns `DomainWorkflowGraph` carrying the forwarded retrieval config |
| INT-08 | `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| INT-09 | `merge_output()` | inner `formatted_answer` -> outer `regulation_answer` AND `result`; `citations`/`status` mapped; changed keys only |
| INT-10 | Config-file fallback | `_parent_config()` never `{}` even with an unreadable `config/config.yaml` |
| — | Live-config probe | a `config/config.yaml` with `top_k: 1` caps the cited passages END-TO-END (the probe query cites multiple passages uncapped first, so the check cannot pass vacuously) |
| INT-11 | e2e happy path | VERIFIED_EXTERNAL invoke -> SUCCESS; `output` = gated formatted answer; PostProcessNode traversed |
| — | input_context bridge e2e | a `category` filter supplied ONLY via `input_context` constrains retrieval through the full nested graph; an invalid `top_k` via `input_context` fails closed (ERROR, empty output) |
| INT-12 | e2e trust denial | ANONYMOUS invoke -> ERROR; empty `output`; PostProcessNode NOT traversed |
| — | JSON-string helpers | `to_json`/`from_json` round-trip; None/malformed handling |

## Marketplace Entry Point — `tests/unit/test_cli_entry_point.py`

| ID | Case | Expected |
|----|------|----------|
| CLI-01 | `cli.py` imports | module loads; `run_agent_marketplace`, `load_agent_config` and `LogisticsRegulationQAAgent` are present |
| CLI-02 | override seam ships empty | `extend_config == {}`; a stray value would silently outrank `config/config.yaml` on the Marketplace path only |
| CLI-03 | the runner receives what the image's CMD would send | executing `cli.py` as `__main__` with the runner replaced captures the call: the graph class, `agent_name`, `namespace`, and every value declared in `config/config.yaml`. Loading the module alone never runs that block, so a wrong class or a dropped config there would otherwise ship unnoticed |

`cli.py` is imported by no other module, so nothing else in the suite would
notice if its import path, graph class or config assembly broke; the image
would build and fail only when the Pod starts. Skipped where the platform
events package is absent.

## 4. Proof-of-Boundary

| ID | Case | Expected |
|----|------|----------|
| PB-IMPORT | `test_import_isolation.py` | no platform-internal SDK import anywhere under `src/` |
| PB-STATE | `test_state_safety.py` | `State` has no credential-named fields and no `BaseModel` / `InvocationContext` annotations |
| PB-6 | `test_pb_invoke_order.py` | full `Graph().invoke()` with `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` (never `for_internal()`) over a valid domain payload -> SUCCESS with outer `node_history` exactly `[InitializeNode, PreProcessNode, LogisticsRegulationSearchGraphNode, PostProcessNode, FinalizeNode]` |
| PB-E2E | `test_invoke_e2e.py` | end-to-end business behaviour through the real ASGI `POST /invoke` (Bearer auth): grounded cited answer from the seeded KB; `input_context` `top_k`/`category` constrain the inner run (bridge proof, non-vacuous); every invalid caller parameter fails closed (`status=error`, no output); no-coverage degrade; disclaimer on every success answer; no credential-shaped content in any rendered answer; wrong token -> generic 401; oversized `input_context` -> 413 |
| PB-5 | `test_state_safety.py` | **Auto-waived — checkpointing disabled** (`config/config.yaml` enables neither `memory_enabled` nor `hitl.enabled`); the conditional gate and the non-lossy traversal helper ship with the stub |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | **Auto-waived — non-HITL** (`config/config.yaml` has no `hitl.enabled: true`); conditional skip-stub retained |
| PB-BOOT | `test_server_boot.py` | `import src.api.server` does not raise; module-level agent is this template's class, compiled; fresh ctor->`compile()` fills the 5 backbone slots; `/health` reports the agent |

> **Gate checklist:** PB-IMPORT, PB-STATE, PB-6, PB-E2E and PB-BOOT are
> mandatory. PB-7 applies only to HITL-enabled templates — this template is
> non-HITL, so PB-7 is **Auto-waived — non-HITL** and its skip must not block
> the gate.

## 5. Test Execution Summary

> **Pending re-run.** The figures below predate the Marketplace entry point work.
> The entry-point test and the PB-5 / PB-6 additions were added after this run and
> have not been executed locally — the framework wheel is not installed in the
> authoring environment. **They are not yet verified anywhere**; this summary is
> updated once a pipeline run has executed them.

- Execution date: 2026-08-25
- Runner: `python -m pytest tests` (full tree, includes
  `tests/proof_of_boundary/`) against `agenticstar-agentcore==1.0.1`
- Total tests: 203
- Pass: 202 / Fail: 0 / Skip: 1 (PB-7 conditional stub — auto-waived, non-HITL)
- Determinism: no LLM, no network; retrieval + answer assembly are rule-based
- `OVERALL: PASS`
