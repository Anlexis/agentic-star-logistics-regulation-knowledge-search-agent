# LOG-C2-005 — Unit Tests: manifest / config consistency
#
# config/agent.yaml is the static AgentRegistry manifest (flat schema: every
# key at ROOT level — identity, entry point, trust level, compile-time
# requires). config/config.yaml carries the runtime parameters; the
# `retrieval` / `llm` blocks there are live configuration, not documentation:
# LogisticsRegulationSearchGraphNode._parent_config() forwards them into the
# inner graph, and the class-name contract requires the declared class to BE
# the src/graph/graph.py agent class. These tests pin manifest <-> config <->
# code consistency so a drift fails fast in CI.
#
# Mirrors docs/03_test_spec.md section 2.8 (CFG-01..CFG-07).
# Deterministic — no LLM, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import LogisticsRegulationQAAgent, LogisticsRegulationSearchGraphNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_template_id_is_consistent(self):
        assert _MANIFEST["id"] == "LOG-C2-005"
        assert _MANIFEST["namespace"] == "log"
        assert _MANIFEST["enabled"] is True

    def test_cfg_02_declared_class_is_the_graph_class(self):
        # Class-name contract: manifest class == graph.py class == server import.
        assert _MANIFEST["class"] == "src.graph.graph.LogisticsRegulationQAAgent"
        module_path, _, class_name = _MANIFEST["class"].rpartition(".")
        assert class_name == LogisticsRegulationQAAgent.__name__
        assert module_path == LogisticsRegulationQAAgent.__module__
        assert _MANIFEST["name"] == LogisticsRegulationQAAgent().name

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "LOG"
        assert _MANIFEST["base_type"] == "RAGAgent"
        assert _MANIFEST["generation_mode"] == "deterministic"

    def test_requires_blocks_match_the_code(self):
        # The pipeline is deterministic: no ctx.secrets.require() call and no
        # LLM client construction anywhere in src/ — both blocks stay empty.
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared

    def test_cfg_05_max_retry_within_framework_ceiling(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # AgentBaseGraph MAX_RETRY_CEILING

    def test_hitl_is_not_enabled(self):
        # This template declares no HITL.
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False


class TestRetrievalBlock:
    def test_cfg_06_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror config/config.yaml — a drift silently
        # changes tuning.
        retrieval = _RUNTIME["retrieval"]
        from src.nodes.retrieve_node import _DEFAULT_RETRIEVAL as retrieve_defaults
        from src.nodes.rerank_filter_node import _DEFAULT_RETRIEVAL as rerank_defaults

        assert retrieval["top_k"] == retrieve_defaults["top_k"] == rerank_defaults["top_k"]
        assert (
            retrieval["score_threshold"] == retrieve_defaults["score_threshold"] == rerank_defaults["score_threshold"]
        )
        assert retrieval["kb_path"] == retrieve_defaults["kb_path"]
        assert retrieval["hybrid_search"] == retrieve_defaults["hybrid_search"]
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_cfg_07_parent_config_forwards_runtime_blocks(self):
        from src.graph.graph import _runtime_config

        cfg = LogisticsRegulationSearchGraphNode(runtime_config=_runtime_config())._parent_config()
        assert cfg["configurable"]["retrieval"] == _RUNTIME["retrieval"]
        assert cfg["configurable"]["llm"] == _RUNTIME["llm"]
        assert cfg["configurable"]["retrieval"], "_parent_config() must never forward an empty retrieval block"

    def test_cfg_07b_runtime_keys_reach_the_forwarded_config(self):
        # timeout_s (file key) travels as timeout_seconds (consumer key);
        # max_retry travels unchanged — declared values must never go dead.
        from src.graph.graph import _runtime_config

        cfg = LogisticsRegulationSearchGraphNode(runtime_config=_runtime_config())._parent_config()
        assert cfg["configurable"]["max_retry"] == _RUNTIME["max_retry"]
        assert cfg["configurable"]["timeout_seconds"] == _RUNTIME["timeout_s"]


class TestSeededKnowledgeBase:
    _REQUIRED_KEYS = {"id", "title", "category", "source", "tags", "content"}

    def test_kb_is_a_well_formed_entry_list(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) >= 5, "seeded KB must carry a usable corpus"
        for entry in entries:
            # "codes" is an OPTIONAL key (only dangerous-goods/customs entries
            # that carry an exact regulation code declare it) — required keys
            # must be present, "codes" may or may not be.
            assert self._REQUIRED_KEYS.issubset(entry.keys())
            assert set(entry.keys()) - self._REQUIRED_KEYS <= {"codes"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids))

    def test_kb_codes_entries_are_lowercase_code_shaped_strings(self):
        # RetrieveNode's hybrid bonus matches a lowercased query token
        # 1:1 against this list — codes must already be lowercase-comparable.
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        coded = [e for e in entries if "codes" in e]
        assert coded, "expected at least one entry with an exact regulation code"
        for entry in coded:
            assert isinstance(entry["codes"], list) and entry["codes"]
            for code in entry["codes"]:
                assert code == code.lower()
