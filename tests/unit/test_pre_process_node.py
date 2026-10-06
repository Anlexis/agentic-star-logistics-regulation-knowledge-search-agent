# LOG-C2-005 — Unit Tests: PreProcessNode (outer pre_process slot)
#
# Invocation canon: every test invokes the node via node(state) —
# BaseNode.__call__ -> trust gate -> PII mask -> execute() -> credential
# scan — never a bare node.execute(state). PreProcessNode requires
# VERIFIED_EXTERNAL, so its behavioural tests build the state at that
# level (the ANONYMOUS rejection lives in test_trust_gate.py).
#
# Input-mask note: this node's own execute() does no additional identifier
# screening (no node-level regex strip) — only the FRAMEWORK input gate masks
# user_input/validated_input before execute() runs (e-mail, SSN/phone/CC digit
# groups, Title-Case name bigrams -> [MASKED]). Intentional-PII tests assert
# the raw identifier is gone and [MASKED] is present.
#
# Mirrors docs/03_test_spec.md section 2.1 (PRE-01..PRE-08).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.pre_process_node
from src.nodes.pre_process_node import PreProcessNode

# Lowercase logistics-regulation phrasing on purpose: PII-free (no Title-Case
# bigram, no @, no digit run), so the framework input mask leaves the payload
# untouched.
_VALID_QUERY = (
    "what packing group and un number applies to lithium batteries by sea " "and what documentation does imdg require"
)


def _make_state(user_input=_VALID_QUERY, **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPreProcessSuccess:
    def test_pre_01_valid_query_accepted(self):
        result = PreProcessNode()(_make_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        assert type(result["status"]) is str  # noqa: E721 — AgentStatus is a str-Enum, so isinstance() cannot catch the enum leaking into State
        assert result["validated_input"] == _VALID_QUERY

    def test_enriched_context_carries_channel(self):
        result = PreProcessNode()(_make_state(input_context={"channel": "web"}))
        assert result["enriched_context"]["channel"] == "web"
        assert result["enriched_context"]["source"] == "LogisticsRegulationQAAgent"

    def test_missing_channel_defaults_to_unknown(self):
        result = PreProcessNode()(_make_state())
        assert result["enriched_context"]["channel"] == "unknown"


class TestPreProcessRejection:
    def test_pre_02_empty_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input=""))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert result["error_log"]
        # No validated_input is produced on the reject path.
        assert "validated_input" not in result

    def test_whitespace_only_is_error(self):
        result = PreProcessNode()(_make_state(user_input="   \n\t "))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")

    def test_pre_03_missing_user_input_is_error(self):
        state = _make_state()
        del state["user_input"]
        result = PreProcessNode()(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")

    def test_non_string_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input={"malicious": "dict"}))
        assert result["status"] == AgentStatus.ERROR.value


class TestPreProcessIdentifierScreen:
    """PRE-04: raw identifiers never survive into validated_input.

    LOG-C2-005's PreProcessNode has no node-level identifier screen — only
    the framework input mask runs.
    """

    def test_email_masked_by_framework_input_gate(self):
        raw = "escalate the customs compliance review to trade.desk@example.com today"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "trade.desk@example.com" not in vi
        assert "[MASKED]" in vi

    def test_grouped_shipment_digits_masked(self):
        # 4-4-4 digit groups match the framework's masked number patterns.
        raw = "shipment 1234 5678 9012 requires an inspection hold"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "1234 5678 9012" not in vi
        assert "[MASKED]" in vi


class TestPreProcessAudit:
    def test_pre_08_domain_audit_payload(self, monkeypatch):
        """The accepted request emits pre_process_complete; the assertion
        targets call.args[1] — the event payload — never the whole call repr."""
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.pre_process_node, "emit_trace_event", spy)
        PreProcessNode()(_make_state())
        events = [call.args[0] for call in spy.call_args_list]
        assert "pre_process_complete" in events
        payload = spy.call_args_list[events.index("pre_process_complete")].args[1]
        assert payload["input_chars"] == len(_VALID_QUERY)
