"""AgentCore Platform v1.0"""

# LOG-C2-005 - InputValidateNode
# Domain node 1: validate the caller's invocation parameters and normalise
# the incoming logistics-regulation question. Two caller-data sources are
# supported, validated identically:
#
#   input_context (dict)  -> {"category": "...", "top_k": N}
#       The platform's structured invocation parameter, bridged into the
#       inner graph by src/graph/context_bridge.py. Wins over the envelope
#       when both supply the same field.
#   input string          -> plain text: the whole string is the search query
#                            JSON envelope {"query": "...", "category": "...",
#                            "top_k": N}: query + structured filters
#
# Validation contract (fail CLOSED):
#   query     free text; whitespace-normalised; hard-capped at 2000 chars.
#   category  optional; must normalise to an inert identifier
#             ([a-z0-9_]{1,32}) - anything else rejects the run.
#   top_k     optional; must be a FINITE integral number in [1, 20] - bools,
#             non-numerics, NaN/Infinity (both parse via float() and arrive
#             via raw JSON), fractional and out-of-range values reject the
#             run. A NaN bound would otherwise compare False everywhere and
#             silently disable the cap it exists to enforce.
#   A rejection terminates the run with status=error and an error_log entry
#   that NAMES the field - the rejected value itself is never echoed into
#   logs, notes, or output. Absent fields simply fall back to the configured
#   defaults.
#
# Wired by the inner graph (DomainWorkflowGraph); on rejection the graph
# routes straight to END (no retrieval, no answer assembly).
# Returns only changed state keys (partial dict).

import json
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import INVALID_VALUE
from src.schemas.state import _finite_in_range, to_json

# Hard cap on the normalised query length (defence-in-depth on input size).
_MAX_QUERY_CHARS = 2000

# Bounds for the caller-supplied top_k override (untrusted numeric guard).
_TOP_K_MIN = 1
_TOP_K_MAX = 20

# Caller-supplied category filter: inert identifier only. The category is
# compared against KB entry categories; free text here is caller-controlled
# content with no legitimate use.
_CATEGORY_RE = re.compile(r"^[a-z0-9_]{1,32}$")

_WHITESPACE_RE = re.compile(r"\s+")


def _validate_top_k(value: Any) -> Tuple[Optional[int], Optional[str]]:
    """Validate the untrusted caller top_k: (parsed, error).

    Returns (None, None) when absent, (int, None) when valid, and
    (None, error-message) on an invalid value. The error message names the
    field and the accepted range - never the supplied value.
    """
    if value is None:
        return None, None
    parsed = _finite_in_range(value, _TOP_K_MIN, _TOP_K_MAX)
    if parsed is None or not float(parsed).is_integer():
        return None, (
            f"InputValidateNode: top_k must be a finite integer between "
            f"{_TOP_K_MIN} and {_TOP_K_MAX} - request rejected."
        )
    return int(parsed), None


def _validate_category(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Validate the untrusted caller category filter: (parsed, error).

    Returns (None, None) when absent, (str, None) when valid, and
    (None, error-message) on an invalid value. Accepts only values that
    normalise (strip + lowercase) to an inert identifier - the error message
    names the field and the accepted pattern, never the supplied value.
    """
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, (
            "InputValidateNode: category must be a string identifier " "(a-z, 0-9, _; max 32 chars) - request rejected."
        )
    normalised = value.strip().lower()
    if not normalised:
        return None, None
    if not _CATEGORY_RE.match(normalised):
        return None, (
            "InputValidateNode: category must be a lowercase identifier "
            "(a-z, 0-9, _; max 32 chars) - request rejected."
        )
    return normalised, None


class InputValidateNode(FunctionNode):
    """Validate caller params and parse the request into a normalised query.

    Input state keys:
        validated_input | user_input: request payload (string or JSON envelope)
        input_context:                structured invocation params (bridged)

    Output state keys (partial dict):
        search_query:  normalised free-text search query
        query_filters: JSON dict {"category": str|None, "top_k": int|None}
        intake_notes:  (when anomalies were seen) JSON list[str]
      or, on a failed validation (fail closed):
        status:    AgentStatus.ERROR.value
        error_log: field-naming rejection message(s)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        emit_progress("Checking the request...")
        raw = state.get("validated_input") or state.get("user_input", "")
        input_context = state.get("input_context") or {}
        notes: List[str] = []
        errors: List[str] = []

        query = ""
        category: Optional[str] = None
        top_k: Optional[int] = None

        # -- input string: plain text or JSON envelope ---------------------
        envelope_category: Any = None
        envelope_top_k: Any = None
        if isinstance(raw, str) and raw.strip():
            payload: Any = None
            text = raw.strip()
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append(
                        "InputValidateNode: JSON-looking input did not parse - " "treated as plain text query."
                    )
            if isinstance(payload, dict):
                query = str(payload.get("query") or payload.get("question") or "")
                envelope_category = payload.get("category")
                envelope_top_k = payload.get("top_k")
            else:
                query = text
        else:
            notes.append("InputValidateNode: empty request - no query to search.")

        # -- structured params: input_context wins over the envelope -------
        raw_category = input_context.get("category")
        if raw_category is None:
            raw_category = envelope_category
        raw_top_k = input_context.get("top_k")
        if raw_top_k is None:
            raw_top_k = envelope_top_k

        category, category_error = _validate_category(raw_category)
        if category_error:
            errors.append(category_error)
        top_k, top_k_error = _validate_top_k(raw_top_k)
        if top_k_error:
            errors.append(top_k_error)

        if errors:
            # Fail CLOSED on the retrieval path: no search runs on a parameter
            # this node could not accept. The run still COMPLETES, carrying the
            # reason, so the caller can correct the value and send the request
            # again on the same conversation.
            # The rejected values are named by FIELD only - never echoed.
            emit_trace_event(
                "input_validate_rejected",
                {"rejected_fields": len(errors)},
                state,
            )
            emit_progress(INVALID_VALUE)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": errors,
            }

        # Normalise whitespace and cap length.
        query = _WHITESPACE_RE.sub(" ", query).strip()
        if len(query) > _MAX_QUERY_CHARS:
            query = query[:_MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {_MAX_QUERY_CHARS} chars.")

        filters = {"category": category, "top_k": top_k}

        # Domain audit: request parsed and normalised.
        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "has_category_filter": category is not None,
                "has_top_k_override": top_k is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
