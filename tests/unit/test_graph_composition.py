# LOG-C2-005 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (LogisticsRegulationQAAgent / Graph)
# end-to-end via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# Mirrors docs/03_test_spec.md section 3 (INT-05..INT-12).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

import src.graph.graph
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    Graph,
    LogisticsRegulationQAAgent,
    LogisticsRegulationSearchGraphNode,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

_DG_QUERY = (
    "What packing group and UN number applies to lithium batteries by sea, " "and what documentation does IMDG require?"
)


def _run(
    user_input: str,
    trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL,
    config: dict | None = None,
) -> dict:
    # config= mirrors every real entry point: AgentRegistry, src/api/server.py and
    # cli.py each read config/config.yaml themselves and hand the result to
    # Graph(config=...). The graph never opens the file itself.
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph(config=config or {}).invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(LogisticsRegulationQAAgent, AgentBaseGraph)

    def test_int_05_graph_alias(self):
        assert Graph is LogisticsRegulationQAAgent

    def test_state_schema_is_state(self):
        assert LogisticsRegulationQAAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = LogisticsRegulationQAAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], LogisticsRegulationSearchGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in LogisticsRegulationQAAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = LogisticsRegulationSearchGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = LogisticsRegulationSearchGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = LogisticsRegulationSearchGraphNode()
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "t", "source": "s"}])
        delta = node.merge_output(
            {},
            {"formatted_answer": "ANSWER", "citations": citations, "status": AgentStatus.SUCCESS.value},
        )
        # The inner formatted_answer surfaces as BOTH regulation_answer and
        # result (PostProcessNode's output gate reads state["result"]).
        assert delta == {
            "regulation_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "status": AgentStatus.SUCCESS.value,
            # Always present so a reason can never be dropped at the boundary.
            "error_code": "",
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert LogisticsRegulationSearchGraphNode.error_strategy == "propagate"
        assert LogisticsRegulationSearchGraphNode.propagate_hitl is False

    def test_int_10_parent_config_never_empty_without_config_file(self, monkeypatch):
        # Even with an unreadable config file the forwarded config carries the
        # fallback retrieval/llm blocks — never {}.
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", pathlib.Path("/nonexistent/config.yaml"))
        cfg = LogisticsRegulationSearchGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/logistics_regulation_kb.json"
        assert cfg["configurable"]["llm"]

    def test_configured_retrieval_values_reach_the_inner_run(self, monkeypatch, tmp_path):
        # The config/config.yaml re-point probe, proven END-TO-END: a
        # declared retrieval value (top_k: 1) must actually constrain the
        # inner pipeline's output — declared configuration is live, not text.
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "max_retry: 3\ntimeout_s: 30\n"
            "retrieval:\n  top_k: 1\n  score_threshold: 0.25\n"
            '  kb_path: "config/kb/logistics_regulation_kb.json"\n'
            "  hybrid_search: true\n",
            encoding="utf-8",
        )
        broad_query = "dangerous goods shipping documentation requirements"
        baseline = _run(broad_query)
        assert "[2]" in baseline.get("output", ""), "probe query must cite multiple passages before the cap is applied"
        import yaml

        result = _run(broad_query, config=yaml.safe_load(cfg_file.read_text(encoding="utf-8")))
        assert result.get("status") == AgentStatus.SUCCESS.value
        output = result.get("output", "")
        assert "[1]" in output, output
        assert "[2]" not in output, f"top_k=1 must cap the citations to one: {output}"


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_DG_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_11_output_is_the_gated_formatted_answer(self):
        output = _run(_DG_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Logistics Regulation Knowledge Base Search Result")
        assert "[1]" in output
        assert "does not constitute legal, customs, or trade-compliance advice" in output

    def test_int_11_e2e_traverses_the_post_process_gate(self):
        history = _run(_DG_QUERY).get("node_history", [])
        for cls_name in ("PreProcessNode", "LogisticsRegulationSearchGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_no_coverage_query_still_terminates_success(self):
        result = _run("quantum telepathy sandwich recipes")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_input_context_category_filter_reaches_the_inner_run(self):
        # Bridge probe, proven END-TO-END through the outer graph: a category
        # filter supplied ONLY via input_context must constrain retrieval
        # inside the inner graph (GraphNode does not forward input_context —
        # src/graph/context_bridge.py carries it).
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="unit-suite")
        result = Graph().invoke(
            "liability limits for lost cargo",
            ctx=ctx,
            input_context={"category": "dangerous_goods"},
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        # Every cited passage must come from the filtered category — the
        # carrier-liability entries may not appear.
        assert "carrier liability" not in result.get("output", "").lower()

    def test_invalid_input_context_param_fails_closed_e2e(self):
        # A NaN top_k supplied via input_context must terminate the run with
        # a field-naming error and no answer (fail closed through the FULL
        # nested graph).
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="unit-suite")
        result = Graph().invoke(_DG_QUERY, ctx=ctx, input_context={"top_k": float("nan")})
        assert result.get("status") == AgentStatus.SUCCESS.value
        # The caller receives the reason as the response body. The internal
        # reason code is not part of the invoke() contract - what crosses the
        # boundary is the sentence saying what to correct.
        assert "could not be accepted" in result["output"], result
        assert "Logistics" not in result["output"], result

    def test_int_12_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """Trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        main slot (its input gate sees status=error and skips the inner graph)
        and routes past post_process to finalize — no domain answer is ever
        produced."""
        result = _run(_DG_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """JSON-string helpers: producers to_json() on write, consumers from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-001", "score": 0.83, "title": "lithium battery packing group"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "dangerous_goods", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
