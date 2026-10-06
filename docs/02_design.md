# Template Design Specification — LOG-C2-005

**Template ID:** LOG-C2-005
**Template Name:** LogisticsRegulationQAAgent
**Category:** Cat 2 (multi-step domain workflow — RAG pattern)
**Industry:** LOG

## Position in AgentCore Architecture

- **Agent Class:** `LogisticsRegulationQAAgent` (alias `Graph`)
- **L1 Base (framework base class):** `AgentBaseGraph` (outer graph) — direct framework inheritance
- **Inner graph base:** `BaseGraph` (framework base class) — `DomainWorkflowGraph`
- **Pattern:** Cat 2 two-layer nested architecture (outer fixed 5-node backbone +
  `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow)
- **Three-Layer Separation:**
  - State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
    structured fields stored as JSON strings via `to_json()` / `from_json()`
  - Node: framework inheritance via `FunctionNode` (override `execute(self, state) -> dict`
    only — no extra parameters; config flows via State, never a `config` arg)
  - Graph: composition (`register_nodes()` for node substitution); outer
    `add_edges()` is NOT overridden

## Purpose

Logistics regulation knowledge-base search agent: customs rules, IATA/IMDG
dangerous-goods handling, carrier compliance, and documentation-requirement
questions are answered over a seeded logistics-regulation knowledge base —
retrieve → rerank/filter → grounded answer with citations + a standing
advisory disclaimer. Scope is regulation Q&A only — this template does not
classify or file customs declarations, execute any action against an
external system, generate structured documents, or run an autonomous loop.
v1 is fully deterministic (hybrid keyword + exact regulation-code retrieval
+ rule-based grounded answer assembly; no live LLM call — see the v1
Implementation Note below).

## Architecture Overview

### Outer backbone (AgentBaseGraph)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max 3)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level | Gate role |
|------|-------|----------------|----------------------|-----------|
| initialize | InitializeNode (framework default) | session_id, trust_level, schema_version | — (framework) | — |
| pre_process | `PreProcessNode` | validate non-empty input → `validated_input` | `TrustLevel.VERIFIED_EXTERNAL` | input trust gate |
| main | `LogisticsRegulationSearchGraphNode` (`GraphNode`) | delegates to inner `DomainWorkflowGraph`; bridges `input_context`; maps inner `formatted_answer` → outer `result` | — (GraphNode delegation) | — (delegates) |
| post_process | `PostProcessNode` | output gate — module-level `_security_gate_output()` scans the rendered answer for credentials → ERROR + sanitised stub across every representation | `TrustLevel.VERIFIED_EXTERNAL` | output gate |
| finalize | FinalizeNode (framework default) | response_metadata, total_time_ms | — (framework) | — |

### Inner graph (DomainWorkflowGraph — BaseGraph, linear)

```
START → input_validate → {route} → retrieve → rerank_filter → generate_answer → output_format → END
                            ↓
                           END   (fail-closed rejection of invalid caller params)
```

All five inner domain nodes declare `required_trust_level = TrustLevel.ANONYMOUS`
(the external trust gate lives on the outer backbone gate node; a stricter inner
level would deny a real VERIFIED_EXTERNAL invoke at runtime).

| Node | Responsibility | required_trust_level | Input State | Output State |
|------|----------------|----------------------|-------------|--------------|
| `InputValidateNode` | Validate the caller's structured params from `input_context` and/or the JSON envelope — `category` locked to `[a-z0-9_]{1,32}`, `top_k` a finite integral 1–20 (bools, NaN/±Infinity, fractional, out-of-range all REJECT the run, naming the field, never echoing the value); normalise whitespace; cap query at 2000 chars | `TrustLevel.ANONYMOUS` | `validated_input` \| `user_input`, `input_context` | `search_query`, `query_filters`, `intake_notes` (or `status=error` + `error_log`) |
| `RetrieveNode` | Deterministic hybrid retrieval over the seeded KB (`config/kb/logistics_regulation_kb.json`): tokenise query, score title/tags/content overlap, add an exact regulation-code match bonus (UN number / HS heading / IATA-IMDG designator), apply category filter | `TrustLevel.ANONYMOUS` | `search_query`, `query_filters`, `retrieval_config` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Rerank candidates (category-match boost), drop entries below `score_threshold`, cap at `top_k` | `TrustLevel.ANONYMOUS` | `retrieved_documents`, `query_filters`, `retrieval_config` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based grounded answer assembly from the ranked KB passages only, with numbered citation markers (v1 deterministic — LLM synthesis seam documented below) | `TrustLevel.ANONYMOUS` | `ranked_documents`, `search_query` | `grounded_answer`, `citations` |
| `OutputFormatNode` | Compose the final answer: body + Sources list + the standing LOG advisory disclaimer (disclaimer is part of this node, NOT post_process) | `TrustLevel.ANONYMOUS` | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Data Flow

```
user_input (+ input_context)
  → PreProcessNode (input trust gate)             → validated_input
  → LogisticsRegulationSearchGraphNode.extract_input → stashes input_context (context bridge)
                                                    → inner DomainWorkflowGraph.invoke(validated_input)
        → input_validate                          → search_query / query_filters (or fail-closed error)
        → retrieve                                → retrieved_documents
        → rerank_filter                           → ranked_documents
        → generate_answer                         → grounded_answer / citations
        → output_format                           → formatted_answer (+ advisory disclaimer)
     get_output() → {formatted_answer, citations, status, error_log, ...}
  → LogisticsRegulationSearchGraphNode.merge_output → result = formatted_answer, regulation_answer
  → PostProcessNode (output gate)                  → formatted_output (gated)
```

Structured invocation parameters arrive on the SDK's first-class
`input_context` dict (`BaseGraph.invoke(user_input, ctx=..., input_context=...)`;
the standalone `/invoke` adapter accepts it as a request field and caps its
serialized size at 256 KiB). `GraphNode.execute()` (SDK 1.0.1) does not
forward `input_context` into `subgraph.invoke()`, so the outer
`extract_input()` stashes it in a ContextVar (`src/graph/context_bridge.py`)
and the inner graph's `_extra_initial_state()` seeds it back into the inner
state — where `InputValidateNode` validates every field. A JSON envelope in
the input string (`{"query": ..., "category": ..., "top_k": ...}`) is
supported for string-only callers and validated identically; `input_context`
wins when both supply the same field.

### Runtime config forwarding (`_parent_config`)

`config/agent.yaml` is the static registry manifest (flat schema: identity,
entry point `class: "src.graph.graph.LogisticsRegulationQAAgent"`, trust
level, compile-time `requires`). Every tunable runtime value lives in
`config/config.yaml` (`max_retry`, `timeout_s`, and the `retrieval` / `llm`
blocks).

`LogisticsRegulationSearchGraphNode._parent_config()` loads
`config/config.yaml` and forwards its keys under `config["configurable"]`
(the `retrieval` / `llm` blocks are never `{}` — module fallbacks cover an
unreadable file; `timeout_s` is renamed to `timeout_seconds`, the key the
graph layer consumes):

```
{"configurable": {"retrieval": {top_k, score_threshold, kb_path, hybrid_search}, "llm": {...}, "max_retry": ..., "timeout_seconds": ...}}
```

`get_subgraph()` passes this into `DomainWorkflowGraph(config=...)`; the inner
graph republishes the `retrieval` block into the inner initial state as the
JSON-string field `retrieval_config` (via `_extra_initial_state()`), so the
declared `top_k` / `score_threshold` / `hybrid_search` are live at runtime.
`RetrieveNode` and `RerankFilterNode` read these settings from the
state-seeded `retrieval_config` field (no `config` parameter on
`execute()`), falling back to module defaults that mirror the
`config/config.yaml` values when unseeded (e.g. direct unit-test
construction).

### State Definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `Optional[str]` | gate-checked query payload | outer |
| `regulation_answer` | `Optional[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `search_query` | `Optional[str]` | normalised search query | inner |
| `query_filters` | `Optional[str]` (JSON) | validated structured params (`category`, `top_k`) | inner |
| `retrieval_config` | `Optional[str]` (JSON) | forwarded `config/config.yaml` `retrieval` block | inner |
| `retrieved_documents` | `Optional[str]` (JSON) | scored KB candidates | inner |
| `ranked_documents` | `Optional[str]` (JSON) | reranked + threshold-filtered passages | inner |
| `grounded_answer` | `Optional[str]` | rule-assembled grounded answer body | inner |
| `citations` | `Optional[str]` (JSON) | `[{ref, id, title, source}]` | inner |
| `formatted_answer` | `Optional[str]` | final answer + sources + advisory disclaimer | inner |
| `intake_notes` | `Optional[str]` (JSON) | validation / parse notes (no PII, no rejected values) | inner |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

**State Constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serialisable types).
- Structured fields (dict / list[dict]) stored as JSON STRINGS via `to_json()` /
  `from_json()` — used consistently by every producer AND consumer (msgpack
  safety).
- Domain fields are `Optional[...]` (valid TypedDict before any node writes).
- `formatted_output` is NOT re-declared (backbone field stays framework-owned).
- No JWT, API keys, credentials, or raw personal identifiers in State.
- `InvocationContext` via `config["configurable"]` only (never in State).
- No Pydantic models / dataclasses / arbitrary Python objects.

## Security Gates

- **Trust gate / input validation:** every node declares
  `required_trust_level` (see tables above); `PreProcessNode`
  (VERIFIED_EXTERNAL) rejects empty / non-string `user_input` before the inner
  workflow runs. The standalone server elevates authenticated Bearer callers to
  VERIFIED_EXTERNAL (`INVOKE_AUTH_TOKEN`).
- **Caller-parameter validation (fail closed):** `InputValidateNode` validates
  every structured caller field against explicit bounds — `category` locked to
  the inert identifier pattern `[a-z0-9_]{1,32}`, `top_k` through a
  finite+bounded parser (`_finite_in_range`) that rejects bools, non-numerics,
  NaN/±Infinity, fractional and out-of-range values. An invalid value
  terminates the run (`status=error`) with an error that NAMES the field —
  rejected values are never echoed into logs, notes, or output. Absent fields
  fall back to the configured defaults.
- **Output gate:** `PostProcessNode` calls the module-level
  `_security_gate_output()` scan from `execute()` — API keys / JWT / Bearer
  tokens / credential assignments in the final rendered answer replace EVERY
  outward representation (`formatted_output`, `result`, `regulation_answer`,
  `citations`) with a sanitised stub, return `AgentStatus.ERROR`, and emit an
  `output_gate_violation` audit event. No `_extra_security_gate_input` /
  `_extra_security_gate_output` instance methods are defined on any node (the
  installed SDK auto-wraps such hooks — prohibited).
- **Audit logging:** every node's `execute()` emits exactly ONE
  domain-specific `emit_trace_event("<node>_complete", {small non-PII payload},
  state)` (free function, positional args) on its success path (plus
  `input_validate_rejected` / `output_gate_violation` on the guard paths).
  Nodes do NOT emit `node_start` / `node_complete` / `node_error` —
  `BaseNode.__call__()` emits those. Domain event names (the names
  operators monitor):
  - `pre_process_complete`
  - `input_validate_complete` / `input_validate_rejected`
  - `retrieve_complete`
  - `rerank_filter_complete`
  - `generate_answer_complete`
  - `output_format_complete`
  - `post_process_complete` / `output_gate_violation`

**Completion is not the same as answering.** A run that ends with
`AgentStatus.SUCCESS` reports that the request was handled safely to a defined
end, not that an answer was produced. A value the caller can correct (an
out-of-contract parameter, an empty or over-long request) ends this way so the
caller receives the reason and can send a corrected request on the same
conversation; terminating instead would end the calling surface's turn and
surface only an exception type, leaving the reason reachable solely from the
audit trail. The reason travels as `error_code` in State, every later domain
node passes through without doing work once it is set, the structured output
fields are withheld, and `PostProcessNode` renders the reason as a static
caller-facing sentence.

Two classes keep terminating, and must not be folded into the above: content
the agent refuses outright (an instruction-override payload — re-sending a
reworded variant is not a correction), and a breach of a contract the caller
cannot influence.

## Advisory Disclaimer

Every answer carries the standing LOG advisory line (informational only, not
legal/customs/trade-compliance advice, verify against primary regulatory
text). It is appended by `OutputFormatNode` as part of the domain output
contract — NOT injected by `post_process` (post_process only gates).

## v1 Implementation Note — LLM synthesis

v1 of this template is **deterministic end-to-end**: retrieval is hybrid
keyword scoring (field-weighted title/tags/content overlap plus an exact
regulation-code match bonus) over the seeded KB, and `GenerateAnswerNode`
assembles the grounded answer rule-based from the ranked passages (lead
sentence + cited passage excerpts). There is NO live LLM call and no LLM
client dependency in v1 — the `config/config.yaml` `llm` block is forwarded
through `_parent_config()` for forward-compatibility but is not consumed by
any v1 node, and no `system_prompt` is read at runtime. The LLM synthesis upgrade
seam is documented in `config/prompts/answer_synthesis_prompt.md`: a v2
`GenerateAnswerNode` swaps the rule-based assembly for an LLM call that
synthesises over the same `ranked_documents` input and emits the same
`grounded_answer` / `citations` state contract, so no other node changes.

## Entry Points

The agent is reachable through three entry points, all of which build the graph
from the same `config/config.yaml`:

| Entry point | Construction | Notes |
|---|---|---|
| Platform registry | `Graph(config=...)` by the registry | Reads `config/config.yaml` itself |
| Standalone HTTP (`src/api/server.py`) | Loads `config/config.yaml`, passes `Graph(config=...)` | Caller-auth boundary; see Security Design |
| Marketplace (`cli.py`) | `run_agent_marketplace(...)` is handed the graph class and the resolved config | The runner constructs the graph itself, so `cli.py` resolves `config/config.yaml` with `load_agent_config()` and passes it in; `extend_config` is the seam for deployment-specific overrides |

`cli.py` sits at the repository root because the deployment image starts it as
`CMD ["python", "cli.py"]`. It adds no business logic: graph construction,
lifecycle, secret provisioning and the invocation loop belong to
`run_agent_marketplace()`.

## Caller-Facing Events

Nodes report progress and rejection reasons to the caller as non-terminal
events, so a caller watching a run sees the pipeline advance instead of a
silent wait, and learns what to change when a request is refused.

- **Progress** — each node reports its phase at the top of `execute()`.
- **Rejection reason** — a node that returns `status: error` sends the reason
  first. It has to happen there: once the run carries an error status the
  framework skips `execute()` on every later node, so no downstream node could
  send it. Wording separates what the caller can fix (missing question,
  oversized request, malformed value) from what they cannot (retrieval or
  output failures), so a caller is not invited into a pointless retry.

Both are best-effort: the emitter is resolved lazily and failures are
swallowed, because reporting must never change the outcome of a run. Messages
are static phase and reason labels — no request value, record value or
internal identifier is ever included, since these events leave the process and
are not covered by the S-3 output gate. Terminal delivery (success/failure)
belongs to the platform runner alone.

## Composition Pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error propagation strategy:** `propagate` (inner errors re-raised as `SubgraphError`).
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; outer pre/post_process run
  at `TrustLevel.VERIFIED_EXTERNAL`.

## Import Isolation Confirmation
- [x] Template does not import the platform-internal SDK.
- [x] Import targets: `framework/` and `shared/` only (no `agents/base/` required).
- [x] No reference-agent class names (`VectorRAGAgent`, `ChatAgent`, …) in any base position.

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step RAG workflow, no autonomous loop |
| Composition pattern | Standalone Cat 1 slots | GraphNode → inner BaseGraph | **GraphNode → inner BaseGraph** | 5-step domain workflow exceeds a single `main` node; nested keeps the outer backbone untouched |
| Answer synthesis | Rule-based assembly | LLM call | **Rule-based (v1)** | Deterministic assembly is testable and dependency-free; v2 swaps the LLM in at the documented seam |
| Retrieval mode | Pure keyword | Hybrid (keyword + exact-code bonus) | **Hybrid (v1)** | Logistics questions frequently cite an exact regulation code (UN number, HS heading, IATA/IMDG designator) that should outrank a loose keyword overlap; still fully deterministic (no embeddings/vector store in v1) |
| KB storage | External vector store | Seeded JSON KB | **Seeded JSON KB (v1)** | Self-contained, deterministic CI; the retrieval contract (`retrieved_documents` JSON) is store-agnostic for a later vector-store upgrade |
