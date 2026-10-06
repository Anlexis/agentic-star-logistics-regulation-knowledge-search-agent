# LOG-C2-005 — Unit Tests: InputValidateNode (inner domain node 1)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# (inner Cat-2 domain node). Payloads are lowercase / PII-free so the
# framework's input mask leaves them untouched.
#
# The caller-parameter contract is fail-CLOSED: category must normalise to an
# inert identifier, top_k must be a finite integral number in [1, 20] — bools,
# non-numerics, NaN/±Infinity (which float() parses and raw JSON delivers),
# fractional and out-of-range values all terminate the run with a
# field-naming error that never echoes the rejected value.
#
# Mirrors docs/03_test_spec.md section 2.2 (VAL-01..VAL-09).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json


def _make_state(payload, **extra) -> dict:
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


def _assert_rejected(result: dict, field: str, raw_value=None) -> None:
    """The fail-closed contract: the retrieval path stays closed, the error
    NAMES the field, and the rejected value is never echoed anywhere in the
    result. The run COMPLETES carrying the reason, so the caller can correct
    the value and send the request again on the same conversation."""
    assert result["status"] == AgentStatus.SUCCESS.value, result
    assert result.get("error_code") == "INVALID_REQUEST", result
    assert any(field in str(e) for e in result["error_log"])
    # Short numerics (0, 21, ...) collide with digits in the static message
    # text; the echo check is meaningful only for distinctive values.
    if raw_value is not None and len(str(raw_value)) >= 3:
        rendered = json.dumps({k: str(v) for k, v in result.items()})
        assert str(raw_value) not in rendered
    assert "search_query" not in result
    assert "query_filters" not in result


class TestPlainTextParsing:
    def test_val_01_plain_text_becomes_query(self):
        result = InputValidateNode()(_make_state("customs entry documentation checklist"))
        assert result["search_query"] == "customs entry documentation checklist"
        filters = from_json(result["query_filters"])
        assert filters == {"category": None, "top_k": None}

    def test_val_02_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state("  customs   entry\n documentation "))
        assert result["search_query"] == "customs entry documentation"

    def test_query_filters_is_json_string(self):
        # Structured State fields travel as JSON strings, never dicts.
        result = InputValidateNode()(_make_state("customs entry documentation"))
        assert isinstance(result["query_filters"], str)
        assert isinstance(from_json(result["query_filters"]), dict)


class TestJsonEnvelopeParsing:
    def test_val_03_envelope_query_category_top_k(self):
        payload = json.dumps({"query": "risk allocation under incoterms", "category": "trade_terms", "top_k": 2})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "risk allocation under incoterms"
        filters = from_json(result["query_filters"])
        assert filters == {"category": "trade_terms", "top_k": 2}

    def test_question_alias_accepted(self):
        payload = json.dumps({"question": "what packing group applies to lithium batteries?"})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "what packing group applies to lithium batteries?"

    def test_category_is_normalised(self):
        payload = json.dumps({"query": "dangerous goods checks", "category": "  Dangerous_Goods "})
        result = InputValidateNode()(_make_state(payload))
        assert from_json(result["query_filters"])["category"] == "dangerous_goods"

    def test_val_04_malformed_json_falls_back_to_plain_text(self):
        payload = "{ this is not valid json but starts like it"
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == payload
        notes = from_json(result.get("intake_notes"), [])
        assert any("did not parse" in n for n in notes)


class TestInputContextParams:
    """Structured invocation params arrive via input_context (bridged into
    the inner state) and are validated identically to the envelope's."""

    def test_input_context_category_and_top_k_are_used(self):
        result = InputValidateNode()(
            _make_state(
                "dangerous goods by sea",
                input_context={"category": "dangerous_goods", "top_k": 3},
            )
        )
        filters = from_json(result["query_filters"])
        assert filters == {"category": "dangerous_goods", "top_k": 3}

    def test_input_context_wins_over_the_envelope(self):
        payload = json.dumps({"query": "dangerous goods by sea", "category": "customs", "top_k": 9})
        result = InputValidateNode()(_make_state(payload, input_context={"category": "dangerous_goods", "top_k": 2}))
        filters = from_json(result["query_filters"])
        assert filters == {"category": "dangerous_goods", "top_k": 2}

    def test_unknown_input_context_keys_are_ignored(self):
        result = InputValidateNode()(_make_state("customs entry documentation", input_context={"unrelated": "x"}))
        assert from_json(result["query_filters"]) == {"category": None, "top_k": None}


class TestTopKFailClosed:
    """VAL-05..07: the caller-supplied top_k is untrusted; every invalid form
    rejects the run (fail closed) with a field-naming, value-free error."""

    @pytest.mark.parametrize(
        "bad_top_k",
        [
            "NaN",
            "Infinity",
            "-Infinity",
            float("nan"),
            float("inf"),
            float("-inf"),
            0,
            -5,
            21,
            99,
            3.5,
            True,
            False,
            "many",
            [4],
            {"n": 4},
        ],
    )
    def test_val_05_invalid_top_k_rejects_via_envelope(self, bad_top_k):
        try:
            payload = json.dumps({"query": "customs entry documentation", "top_k": bad_top_k})
        except (TypeError, ValueError):
            pytest.skip("not JSON-serialisable")
        result = InputValidateNode()(_make_state(payload))
        _assert_rejected(result, "top_k", bad_top_k)

    @pytest.mark.parametrize(
        "bad_top_k",
        ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), 0, 21, 3.5, True, "many"],
    )
    def test_val_06_invalid_top_k_rejects_via_input_context(self, bad_top_k):
        result = InputValidateNode()(_make_state("customs entry documentation", input_context={"top_k": bad_top_k}))
        _assert_rejected(result, "top_k", bad_top_k)

    def test_val_07_integral_float_top_k_is_accepted(self):
        result = InputValidateNode()(_make_state("customs entry documentation", input_context={"top_k": 4.0}))
        assert from_json(result["query_filters"])["top_k"] == 4


class TestCategoryFailClosed:
    @pytest.mark.parametrize(
        "bad_category",
        [
            "not an identifier!",
            "a" * 33,
            "drop table; --",
            "テスト",
            123,
            ["dangerous_goods"],
            {"c": 1},
        ],
    )
    def test_invalid_category_rejects(self, bad_category):
        result = InputValidateNode()(
            _make_state("customs entry documentation", input_context={"category": bad_category})
        )
        _assert_rejected(result, "category", bad_category)

    def test_blank_category_is_treated_as_absent(self):
        result = InputValidateNode()(_make_state("customs entry documentation", input_context={"category": "  "}))
        assert from_json(result["query_filters"])["category"] is None

    def test_both_fields_invalid_names_both(self):
        result = InputValidateNode()(
            _make_state(
                "customs entry documentation",
                input_context={"category": "bad category!", "top_k": "NaN"},
            )
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        joined = " ".join(result["error_log"])
        assert "category" in joined and "top_k" in joined


class TestSizeAndEmptyGuards:
    def test_val_08_oversize_query_is_truncated(self):
        payload = "regulation " * 300  # ~3300 chars after collapse
        result = InputValidateNode()(_make_state(payload))
        assert len(result["search_query"]) == 2000
        notes = from_json(result.get("intake_notes"), [])
        assert any("truncated" in n for n in notes)

    def test_val_09_empty_request_yields_note_not_error(self):
        result = InputValidateNode()(_make_state(""))
        assert result["search_query"] == ""
        notes = from_json(result.get("intake_notes"), [])
        assert any("empty request" in n for n in notes)
