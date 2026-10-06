# LOG-C2-005 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# RULES B6 / C2-retired (2026-07-27): execute(self, state) takes NO config
# parameter — every config knob below is exercised by SEEDING the state field
# retrieval_config (the same field DomainWorkflowGraph._extra_initial_state()
# republishes from the manifest), never a direct execute(state, config=...)
# call.
#
# Mirrors docs/03_test_spec.md section 2.3 (RET-01..RET-09).
# Deterministic — hybrid (keyword + exact regulation-code) scoring over the
# seeded config/kb/logistics_regulation_kb.json; no LLM, no network.
# framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_DG_QUERY = (
    "what packing group and un number applies to lithium batteries by sea " "and what documentation does imdg require"
)
_CUSTOMS_QUERY = "what documents are needed for a standard customs entry for an international shipment"


def _make_state(query=_DG_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_dangerous_goods_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the lithium-battery/IMDG query"
        assert docs[0]["id"] == "kb-001"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {"id", "title", "category", "source", "score", "excerpt"}
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveFilters:
    def test_ret_04_category_filter_restricts_pool(self):
        # Unfiltered this query pulls candidates from customs / import_compliance
        # / carrier_liability / compliance; filtering to "customs" keeps only
        # the two customs-category entries.
        unfiltered = from_json(RetrieveNode()(_make_state(query=_CUSTOMS_QUERY))["retrieved_documents"])
        assert {d["category"] for d in unfiltered} != {"customs"}, "fixture must span >1 category unfiltered"

        state = _make_state(
            query=_CUSTOMS_QUERY,
            query_filters=to_json({"category": "customs", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "customs category has seeded entries"
        assert {d["category"] for d in docs} == {"customs"}
        assert docs[0]["id"] == "kb-002"

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []


class TestRetrieveHybridCodeMatch:
    """RULES note (2026-07-27): cover the hybrid exact regulation-code match
    bonus (UN numbers / HS headings / IATA-IMDG designators) — the field that
    differentiates this template's RetrieveNode from a plain keyword scorer.
    """

    def test_ret_09_code_only_query_hits_via_hybrid_bonus(self):
        # "un3480" alone has no keyword overlap with any KB title/tag/content
        # token (the source text spells it "UN 3480", two separate tokens) —
        # only the exact-code bonus against kb-001's codes list can score it.
        state = _make_state(query="un3480")
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert [d["id"] for d in docs] == ["kb-001"]
        assert docs[0]["score"] == 0.5

    def test_ret_09_hybrid_disabled_finds_nothing_for_the_same_query(self):
        state = _make_state(
            query="un3480",
            retrieval_config=to_json({"hybrid_search": False}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs == [], "without hybrid_search the code-only query has no keyword overlap"


class TestRetrieveConfigPrecedence:
    """Config plumbing (state-seeded only — RULES B6 / C2-retired): state
    retrieval_config > module defaults. Exercised via node(state), never a
    direct execute(state, config=...) call.
    """

    def test_ret_06_state_kb_path_override_degrades_gracefully(self):
        state = _make_state(retrieval_config=to_json({"kb_path": "config/kb/does_not_exist.json"}))
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_07_malformed_state_top_k_falls_back_to_default(self):
        # A non-numeric top_k in the state-seeded retrieval_config must not
        # raise — it falls back to the module default (int() -> ValueError
        # caught) rather than TypeError-ing the whole node call.
        state = _make_state(retrieval_config=to_json({"top_k": "not-a-number"}))
        result = RetrieveNode()(state)
        docs = from_json(result["retrieved_documents"])
        assert docs[0]["id"] == "kb-001"


class TestRetrieveNotesAccumulation:
    def test_ret_08_notes_append_never_clobber(self):
        state = _make_state(
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_config=to_json({"kb_path": "config/kb/bogus.json"}),
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2
