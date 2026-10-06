"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never Pydantic BaseModel. LangGraph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption.  Extend AgentState with agent-specific fields only.  Do NOT add
# credentials, secrets, or Pydantic models.
#
# msgpack safety: structured fields (dict / list[dict]) are stored as JSON
# STRINGS, not bare Python containers - a bare dict/list in a checkpointed
# State field corrupts on round-trip. Producers serialize with to_json() on
# write; consumers deserialize with from_json() on read.
#
# LOG-C2-005 - Logistics Regulation Q&A Agent (Cat 2 RAG).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# PII / confidentiality note: PreProcessNode rejects empty/invalid input
# before any field is written to State. Only the normalised search query, KB
# passage summaries, and the final grounded answer are persisted - never raw
# caller identifiers.

import json
import math
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def _finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse a caller-controlled numeric: FINITE float within [lo, hi], else None.

    Rejects bools, non-numerics, and - critically - non-finite values: float()
    happily parses "NaN"/"Infinity" (and Python's json accepts bare NaN in
    request bodies), and IEEE NaN comparisons are always False, which turns a
    bounds check into silent FAIL-OPEN. Every caller-supplied number must
    come through here (or an equivalent explicit finite check).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for LOG-C2-005.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are Optional so the TypedDict is valid at graph
    initialisation, before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / LogisticsRegulationSearchGraphNode.merge_output
    # ------------------------------------------------------------------

    # Identifier-stripped, validated query payload produced by PreProcessNode.
    # Raw input is NOT persisted beyond PreProcessNode.
    validated_input: Optional[str]

    # Final logistics-regulation search answer, mapped from the inner graph's
    # formatted_answer output via merge_output.
    regulation_answer: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text search query (whitespace-collapsed, length-capped).
    search_query: Optional[str]

    # JSON STRING (to_json) of validated structured query params. Deserialised
    # dict shape: {"category": str | None, "top_k": int | None}.
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: Optional[str]

    # config/config.yaml `retrieval` block forwarded by
    # LogisticsRegulationSearchGraphNode._parent_config() ->
    # DomainWorkflowGraph._extra_initial_state().
    # JSON STRING (to_json) of {"top_k": int, "score_threshold": float,
    # "kb_path": str, "hybrid_search": bool}. Consumers (RetrieveNode,
    # RerankFilterNode) read it back via from_json().
    retrieval_config: Optional[str]

    # RetrieveNode output
    # JSON STRING (to_json) of scored KB candidates. Deserialised shape:
    # list[dict], each entry {"id": str, "title": str, "category": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: Optional[str]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: Optional[str]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers.
    grounded_answer: Optional[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "title": str, "source": str}.
    # Consumers (OutputFormatNode) read it back via from_json().
    citations: Optional[str]

    # OutputFormatNode output
    # Final formatted answer (body + sources + advisory disclaimer). Written by
    # OutputFormatNode; surfaced to the outer graph via get_output() ->
    # merge_output().
    formatted_answer: Optional[str]

    # Validation / parse notes accumulated during intake (no PII, and never a
    # rejected caller value - notes name fields, not contents).
    # JSON STRING (to_json) of list[str].
    intake_notes: Optional[str]

    # ------------------------------------------------------------------
    # Degraded completion marker
    # ------------------------------------------------------------------

    # Set when the run completes WITHOUT producing an answer because the
    # caller's request could not be accepted as written - a rejection the
    # caller can correct and retry. The run still completes: no retrieval is
    # performed, no answer is assembled, and the domain audit event for the
    # rejection is still emitted. Carrying this as a completion marker rather
    # than a terminal error is what lets the caller see the reason and send a
    # corrected request on the same conversation.
    #
    # Content the agent refuses outright, and a breach of a contract the
    # caller cannot influence, are NOT reported here - those stay terminal so
    # they are not mistaken for something a reworded request would get past.
    #
    # Once set, every later domain node passes through without doing work, and
    # the value is carried across the inner/outer boundary by get_output() and
    # merge_output().
    error_code: Optional[str]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
