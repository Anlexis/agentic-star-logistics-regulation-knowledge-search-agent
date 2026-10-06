"""AgentCore Platform v1.0"""

# Output security gate: this node calls the MODULE-LEVEL
# _security_gate_output() scan from execute() itself. The output string is
# scanned for disallowed content (API keys, JWT tokens, Bearer tokens, raw
# credential assignments) — generic, domain-agnostic patterns. On a
# violation, EVERY outward representation of the answer carried by State
# (result, formatted_output, regulation_answer, citations) is replaced with
# a sanitised stub and ERROR status is returned, with an audit event — the
# blocked content must not survive under any other field name. No
# _extra_security_gate_input/_output instance methods are defined on this
# node (the framework auto-wraps such hooks).
#
# This is an outer backbone gate slot — the manifest declares
# required_trust_level: "VERIFIED_EXTERNAL" (config/agent.yaml).

import logging
import re
from typing import Any, ClassVar, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG
from src.services.progress import emit_progress

logger = logging.getLogger(__name__)

# Disallowed content patterns. Each tuple: (name, compiled regex) —
# order matters (most specific first).
_DISALLOWED_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # API key patterns: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # JWT: three base64url segments separated by dots
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # Bearer token in Authorization-like context
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE)),
    # Credential assignment patterns
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

_SANITISED_STUB = (
    "[OUTPUT BLOCKED by the output security gate - disallowed content "
    "detected. Review the generated output and retry without "
    "credential-like strings.]"
)


def _security_gate_output(content: str) -> Optional[str]:
    """Run the output content gate.

    Returns the name of the first matched violation, or None if clean.
    """
    for name, pattern in _DISALLOWED_PATTERNS:
        if pattern.search(content):
            return name
    return None


# Caller-facing wording for a run that completed without an answer. The marker
# is an internal reason code; this maps it to the sentence the caller sees.
# Static sentences only - no request value is ever substituted, so nothing the
# caller sent can be reflected back through this path.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Format and finalize the output, behind the output security gate."""

    # Explicit by design, not inherited implicitly. Outer backbone gate
    # slot — matches the manifest's declared required_trust_level
    # (config/agent.yaml).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        emit_progress("Finalising the response...")

        # The run completed without an answer because the request could not be
        # accepted as written. Report the reason as the response: the caller
        # needs to know what to change, and an empty body would leave them with
        # nothing. Status stays SUCCESS - the run did what it could with the
        # request it was given, and the caller can correct it and send again on
        # the same conversation.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "formatted_output": message,
                "result": message,
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
            }

        result = state.get("result", "")

        if not result or not str(result).strip():
            # No result to gate — forward as-is (non-fatal).
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        # The rendered answer already embeds the citation list, so scanning
        # the single rendered string covers everything the caller sees; on a
        # violation the unrendered representations are replaced too, so the
        # blocked content does not survive in State under another name.
        violation = _security_gate_output(str(result))
        if violation:
            logger.error(
                "PostProcessNode: OUTPUT BLOCKED - violation type: %s",
                violation,
            )
            emit_trace_event(
                "output_gate_violation",
                {"violation_type": violation},
                state,
            )
            blocked: dict[str, Any] = {
                "formatted_output": _SANITISED_STUB,
                "result": _SANITISED_STUB,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: output blocked - " f"disallowed content detected ({violation})"],
            }
            # Replace every other outward representation of the answer.
            if state.get("regulation_answer"):
                blocked["regulation_answer"] = _SANITISED_STUB
            if state.get("citations"):
                blocked["citations"] = None
            return blocked

        # Clean — domain audit: a finalized output was emitted.
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(str(result))},
            state,
        )

        return {
            "formatted_output": result,
            "status": AgentStatus.SUCCESS.value,
        }
