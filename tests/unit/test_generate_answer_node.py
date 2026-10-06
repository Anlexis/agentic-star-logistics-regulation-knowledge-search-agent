# LOG-C2-005 — Unit Tests: GenerateAnswerNode (inner domain node 4)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# The grounded answer / citations are DOMAIN fields (not input-mask targets), so
# Title-Case KB titles inside them are safe to assert on.
#
# Mirrors docs/03_test_spec.md section 2.5 (GEN-01..GEN-05).
# Deterministic — rule-assembled from ranked_documents only (grounded by
# construction; no LLM, no network). framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json


def _ranked(*entries):
    return to_json(list(entries))


def _doc(doc_id, title, excerpt, source="seeded kb"):
    return {
        "id": doc_id,
        "title": title,
        "category": "dangerous_goods",
        "source": source,
        "score": 0.9,
        "excerpt": excerpt,
    }


def _make_state(ranked_documents, query="lithium battery packing group", **extra) -> dict:
    state = {
        "ranked_documents": ranked_documents,
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestGroundedAnswer:
    def test_gen_01_answer_carries_numbered_citation_markers(self):
        ranked = _ranked(
            _doc(
                "kb-001",
                "UN number and packing group for lithium batteries carried by sea (IMDG)",
                "lithium-ion cells and batteries packed by themselves are UN 3480.",
            ),
            _doc(
                "kb-003",
                "IATA dangerous goods classification and Shipper's Declaration for air transport",
                "dangerous goods tendered for air transport must be classified.",
            ),
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        answer = result["grounded_answer"]
        assert "[1] UN number and packing group for lithium batteries carried by sea (IMDG):" in answer
        assert "[2] IATA dangerous goods classification and Shipper's Declaration for air transport:" in answer

    def test_gen_02_lead_sentence_quotes_the_query(self):
        ranked = _ranked(
            _doc("kb-001", "UN number and packing group for lithium batteries carried by sea (IMDG)", "excerpt.")
        )
        result = GenerateAnswerNode()(_make_state(ranked, query="lithium battery packing group"))
        assert 'the question: "lithium battery packing group"' in result["grounded_answer"]

    def test_gen_03_citations_mirror_ranked_order(self):
        ranked = _ranked(
            _doc(
                "kb-001",
                "UN number and packing group for lithium batteries carried by sea (IMDG)",
                "a.",
                source="IMDG Code, Class 9 lithium battery provisions summary",
            ),
            _doc("kb-003", "IATA dangerous goods classification and Shipper's Declaration for air transport", "b."),
        )
        citations = from_json(GenerateAnswerNode()(_make_state(ranked))["citations"])
        assert [c["ref"] for c in citations] == [1, 2]
        assert [c["id"] for c in citations] == ["kb-001", "kb-003"]
        assert citations[0]["source"] == "IMDG Code, Class 9 lithium battery provisions summary"

    def test_citations_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        ranked = _ranked(
            _doc("kb-001", "UN number and packing group for lithium batteries carried by sea (IMDG)", "a.")
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        assert isinstance(result["citations"], str)

    def test_gen_04_answer_is_grounded_in_ranked_passages_only(self):
        ranked = _ranked(
            _doc(
                "kb-001",
                "UN number and packing group for lithium batteries carried by sea (IMDG)",
                "packing group II applies to lithium-ion cells packed alone.",
            )
        )
        answer = GenerateAnswerNode()(_make_state(ranked))["grounded_answer"]
        # Every content line traces to the single ranked passage.
        assert "packing group II applies to lithium-ion cells packed alone." in answer
        assert "[2]" not in answer


class TestNoCoverage:
    def test_gen_05_empty_ranked_set_yields_no_coverage_answer(self):
        result = GenerateAnswerNode()(_make_state(_ranked()))
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []

    def test_missing_ranked_field_is_treated_as_no_coverage(self):
        state = _make_state(None)
        del state["ranked_documents"]
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
