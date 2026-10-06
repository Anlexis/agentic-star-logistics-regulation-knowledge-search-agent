"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus.<X>.value strings for status assignments
#  - Read input_context via state.get("input_context", {}) — read-only
#  - Never import from mediator/, api/, or other agents
#
# This is an outer backbone gate slot — the manifest declares
# required_trust_level: "VERIFIED_EXTERNAL" (config/agent.yaml), so this node
# gates external callers before the inner domain workflow runs. Field-level
# validation of the caller's structured parameters happens in the inner
# graph's InputValidateNode.

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT


class PreProcessNode(FunctionNode):
    """Validate and enrich incoming input before main processing."""

    # Explicit by design, not inherited implicitly. Outer backbone gate
    # slot — matches the manifest's declared required_trust_level
    # (config/agent.yaml).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        emit_progress("Checking the request...")
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only

        if not user_input or not user_input.strip():
            # Nothing to work with, but the caller can simply send a question
            # and try again - so the run completes carrying the reason rather
            # than terminating. Terminating would end the caller's turn and
            # surface only an exception type, leaving the reason reachable
            # solely from the audit trail.
            emit_progress(EMPTY_INPUT)
            emit_trace_event("pre_process_declined", {"reason": "EMPTY_INPUT"}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        validated_input = user_input.strip()

        # Domain audit: a request was accepted for processing.
        emit_trace_event(
            "pre_process_complete",
            {"input_chars": len(validated_input)},
            state,
        )

        return {
            "validated_input": validated_input,
            "enriched_context": {
                "source": "LogisticsRegulationQAAgent",
                "channel": input_context.get("channel", "unknown"),
            },
            "status": AgentStatus.SUCCESS.value,
        }
