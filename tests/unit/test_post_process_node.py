# LOG-C2-005 — Unit Tests: PostProcessNode (outer post_process slot; output gate)
#
# Invocation canon: node(state) via BaseNode.__call__. PostProcessNode is the
# second outer gate slot and requires VERIFIED_EXTERNAL (like PreProcessNode),
# so its behavioural tests build the state at that level; the ANONYMOUS
# rejection lives in test_trust_gate.py.
#
# Gate layering: the node's own module-level _security_gate_output() scan runs
# INSIDE execute() and replaces a violating answer with the sanitised stub
# (returned dict — no exception). The framework's FunctionNode credential
# scan then sees only the clean stub. Intentional-credential tests assert the
# raw secret never survives into ANY outward representation — formatted_output,
# result, regulation_answer, or citations.
#
# Mirrors docs/03_test_spec.md section 2.7 (POST-01..POST-06).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import PostProcessNode

_CLEAN_REPORT = (
    "# Logistics Regulation Knowledge Base Search Result\n\n"
    "[1] lithium-ion batteries packed by themselves are assigned UN 3480.\n"
)

# JWT-shaped token built at runtime so no credential-shaped literal ever sits
# in the repository (CI credential-scan hygiene).
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        assert type(result["status"]) is str  # noqa: E721 — AgentStatus is a str-Enum, so isinstance() cannot catch the enum leaking into State
        assert result["formatted_output"] == _CLEAN_REPORT

    def test_post_02_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestPostProcessOutputGate:
    def _assert_blocked(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        # The raw secret must not survive into ANY surfaced field.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert "[OUTPUT BLOCKED by the output security gate" in result["formatted_output"]

    def test_post_03_api_key_is_blocked(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"# Report\n\n<!-- debug api_key={secret} -->\n"))
        self._assert_blocked(result, secret)

    def test_post_04_credential_assignment_is_blocked(self):
        secret = "password=super_secret_value_123"
        result = PostProcessNode()(_make_state(f"# Report\n\ninternal note: {secret}\n"))
        self._assert_blocked(result, "super_secret_value_123")

    def test_post_05_jwt_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\nsession token {_FAKE_JWT}\n"))
        self._assert_blocked(result, _FAKE_JWT)

    def test_post_06_bearer_token_is_blocked(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(f"# Report\n\nauthorization: {secret}\n"))
        self._assert_blocked(result, secret)


class TestEveryRepresentationIsReplaced:
    """The gate's invariant covers every outward representation of the
    answer, not only the field it happens to scan."""

    def test_violation_replaces_regulation_answer_and_citations(self):
        secret = "sk-ABCDEF0123456789abcdef"
        dirty = f"# Report\n\n<!-- debug api_key={secret} -->\n"
        result = PostProcessNode()(
            _make_state(
                dirty,
                regulation_answer=dirty,
                citations='[{"ref": 1, "id": "kb-001"}]',
            )
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert secret not in str(result.get("regulation_answer", ""))
        assert result["citations"] is None

    def test_clean_output_leaves_other_representations_untouched(self):
        result = PostProcessNode()(_make_state(_CLEAN_REPORT, regulation_answer=_CLEAN_REPORT))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "regulation_answer" not in result  # only changed keys returned
