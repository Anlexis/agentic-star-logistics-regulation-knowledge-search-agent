"""AgentCore Platform v1.0"""

# LOG-C2-005 - DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the full logistics-regulation KB search domain workflow:
#
#   START -> input_validate -> {route} -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#                    |
#                    -> END   (fail-closed rejection of invalid caller params)
#
# Called by LogisticsRegulationSearchGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with LogisticsRegulationSearchGraphNode.merge_output()
#   - No platform-internal SDK imports
#   - Not placed under src/subagents/

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for LOG-C2-005.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by LogisticsRegulationSearchGraphNode.get_subgraph() in graph.py,
    which passes the config/config.yaml-derived config (`_parent_config()`)
    into the ctor.

    Pipeline:
        START
          -> input_validate  (InputValidateNode)  - validate caller params + normalise the query
          -> retrieve        (RetrieveNode)       - hybrid-score the seeded KB
          -> rerank_filter   (RerankFilterNode)   - boost / threshold / top_k cut
          -> generate_answer (GenerateAnswerNode) - grounded answer + citations
          -> output_format   (OutputFormatNode)   - final format + advisory disclaimer
          -> END

    input_validate routes to END instead of retrieve when a caller-supplied
    parameter fails validation (fail-closed - the run terminates with
    status=error and a field-naming error_log entry).

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns - not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "log_c2_005_logistics_regulation_search_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The forwarded `retrieval` block (top_k / score_threshold / kb_path /
        hybrid_search) is read per-call by the domain nodes with safe
        defaults, so absence is non-fatal. Validation is permissive here
        rather than raising ConfigError.
        """
        pass

    # -- Config forwarding into state (config/config.yaml -> inner nodes) -------

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the inner state with the retrieval config and caller context.

        LogisticsRegulationSearchGraphNode._parent_config() forwards the
        config/config.yaml `retrieval` + `llm` blocks under
        config["configurable"]; this hook makes the `retrieval` block
        reachable by the domain nodes at runtime as the JSON-string state
        field `retrieval_config` (msgpack-safe convention: JSON string, not a
        bare dict). RetrieveNode / RerankFilterNode read this field (config
        flows in via State only - no config parameter on execute()).

        It also seeds `input_context` from the context bridge:
        GraphNode.execute() (SDK 1.0.1) does not forward the outer state's
        input_context into subgraph.invoke(), so without this hook the
        caller's structured invocation parameters (category / top_k) would
        never reach InputValidateNode - see src/graph/context_bridge.py.
        """
        retrieval = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        return {
            "retrieval_config": to_json(retrieval),
            "input_context": get_caller_input_context(),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments - FunctionNode
        subclasses take no __init__; config flows in per-call via state
        (execute(self, state) only). Every key registered here is referenced
        in add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the logistics-regulation KB search domain topology.

        Linear pipeline with a single fail-closed branch: after
        input_validate, route() sends a run that rejected a caller parameter
        (status=error) straight to END - no retrieval, no answer assembly -
        so a rejected value can never influence the produced output. Every
        other step passes its partial-dict output into the shared State.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_conditional_edges("input_validate", self.route, {"retrieve": "retrieve", END: END})
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing after input_validate (fail-closed rejection).

        A run whose caller parameters failed validation carries status=error
        and terminates immediately; every other run continues into retrieval.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "retrieve"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by LogisticsRegulationSearchGraphNode.merge_output()
        in graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output()  emits: "formatted_answer", "citations", "status", ...
            Outer merge_output() reads: sub_result.get("formatted_answer"),
                                        sub_result.get("citations"),
                                        sub_result.get("status")

        error_log carries the inner run's recorded errors (e.g. the
        fail-closed rejection message naming an invalid caller field) so
        merge_output() can surface them on the outer state. Additional fields
        (intake_notes, trace_id, correlation_id, node_history) are surfaced
        for observability / downstream extension.
        """
        return {
            "formatted_answer": state.get("formatted_answer"),
            "citations": state.get("citations"),
            "status": state.get("status"),
            # Carried explicitly: the boundary only moves the keys named here,
            # so a run that completed without an answer would otherwise arrive
            # at the outer graph indistinguishable from one that answered.
            "error_code": state.get("error_code"),
            "error_log": state.get("error_log", []),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
