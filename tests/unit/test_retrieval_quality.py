# LOG-C2-005 — Unit Tests: retrieval quality over the seeded KB
#
# Golden-query suite: drives the REAL inner retrieval chain
# (InputValidateNode -> RetrieveNode -> RerankFilterNode) via node(state) /
# __call__ (ANONYMOUS inner nodes) against
# config/kb/logistics_regulation_kb.json and pins the expected top hit per
# domain query — one golden query per seeded KB entry (kb-001..kb-010). The
# scorer is deterministic (keyword field-weights + exact regulation-code
# bonus, stable tie-break), so exact top-1 assertions are safe and catch
# KB / scorer / threshold regressions.
#
# Mirrors docs/03_test_spec.md section 2.9 (QUAL-01..QUAL-07).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
import pathlib

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB_IDS = {
    entry["id"]
    for entry in json.loads((_ROOT / "config" / "kb" / "logistics_regulation_kb.json").read_text(encoding="utf-8"))
}

_DEFAULT_SCORE_THRESHOLD = 0.25  # mirrors config/agent.yaml retrieval block


def _search(payload: str) -> list[dict]:
    """Run the real inner retrieval chain and return the surviving passages."""
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "quality-session",
        "execution_time": {},
    }
    state.update(InputValidateNode()(state))
    state.update(RetrieveNode()(state))
    state.update(RerankFilterNode()(state))
    return from_json(state["ranked_documents"], [])


# (query, expected top-1 KB entry id) — one query per seeded KB entry,
# verified against the deterministic scorer.
_GOLDEN_QUERIES = [
    (
        "what packing group and un number applies to lithium batteries by sea "
        "and what documentation does imdg require",
        "kb-001",
    ),
    ("what documents are needed for a standard customs entry for an international shipment", "kb-002"),
    ("shipper declaration for dangerous goods class required for air transport", "kb-003"),
    ("who bears risk and cost under fob versus cif incoterms", "kb-004"),
    ("what hs heading and duty rate applies to network switching apparatus", "kb-005"),
    ("restricted party sanctions screening before booking a shipment", "kb-006"),
    ("carrier liability limit for lost cargo under the montreal convention", "kb-007"),
    ("pre-trip inspection requirement for temperature controlled reefer cargo", "kb-008"),
    ("import license requirement for controlled chemicals at destination country", "kb-009"),
    ("rules of origin test to claim preferential duty under a free trade agreement", "kb-010"),
]


class TestGoldenQueries:
    @pytest.mark.parametrize(("query", "expected_id"), _GOLDEN_QUERIES)
    def test_qual_01_top_hit_per_golden_query(self, query, expected_id):
        kept = _search(query)
        assert kept, f"no passage cleared the relevance floor for: {query!r}"
        assert kept[0]["id"] == expected_id

    def test_qual_02_all_survivors_clear_the_relevance_floor(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["score"] >= _DEFAULT_SCORE_THRESHOLD

    def test_qual_03_survivor_ids_exist_in_the_seeded_kb(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["id"] in _KB_IDS


class TestPrecision:
    def test_qual_04_dangerous_goods_query_keeps_only_the_top_entry(self):
        # Off-topic passages score below the floor and are cut — precision, not
        # just recall.
        kept = _search(_GOLDEN_QUERIES[0][0])
        assert [d["id"] for d in kept] == ["kb-001"]

    def test_qual_05_category_filter_restricts_to_that_category(self):
        payload = json.dumps(
            {"query": "what hs heading and duty rate applies to network switching apparatus", "category": "customs"}
        )
        kept = _search(payload)
        assert kept, "customs category carries seeded entries"
        assert {d["category"] for d in kept} == {"customs"}
        assert kept[0]["id"] == "kb-005"


class TestNoCoverage:
    def test_qual_06_out_of_domain_query_yields_no_survivors(self):
        assert _search("quantum telepathy sandwich recipes") == []

    def test_qual_07_no_coverage_produces_the_escalation_answer(self):
        state = {
            "ranked_documents": "[]",
            "search_query": "quantum telepathy sandwich recipes",
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "quality-session",
            "execution_time": {},
        }
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []
