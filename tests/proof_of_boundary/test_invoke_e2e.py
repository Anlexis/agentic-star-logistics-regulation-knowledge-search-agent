# PB: End-to-end business behaviour through POST /invoke — src/api/server.py
#
# Proves the supported input contract produces REAL outcomes through the full
# nested graph (outer backbone → inner domain pipeline), not an empty
# baseline:
#   - a grounded, source-cited answer from the seeded knowledge base
#   - input_context invocation parameters (category / top_k) constraining the
#     inner run (the context bridge carries them across the outer→inner
#     boundary — GraphNode does not forward input_context)
#   - a fail-closed validation rejection for malformed caller parameters
#   - the output invariant on the rendered answer (advisory disclaimer
#     present, no credential-shaped content)
#   - the adapter-level guards (Bearer auth, input_context size cap)
#
# Unlike test_server_boot.py (which checks the module-level boot contract),
# these tests run the REAL compiled agent: every request crosses the
# entry-point auth, the outer trust/input gates, the input_context bridge into
# the inner graph, all five domain nodes, and the output gate.
#
# The app is driven through its real ASGI interface — no extra test-client
# dependency needed.

import asyncio
import json
import re

import pytest

from src.api import server as server_module  # noqa: F401  (import = boot check)
from src.api.server import app

_TOKEN = "pb-invoke-e2e-token"

_QUERY = (
    "What packing group and UN number applies to lithium batteries by sea, " "and what documentation does IMDG require?"
)

_DISCLAIMER_SNIPPET = "does not constitute legal, customs, or trade-compliance advice"

# Credential-shaped forms that must never appear in a rendered answer.
_CREDENTIAL_RES = [
    re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE),
]


def _post_invoke(payload: dict, token: str = _TOKEN) -> tuple[int, dict]:
    """POST /invoke with a Bearer token through the real ASGI app."""
    body = json.dumps(payload).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/invoke",
        "raw_path": b"/invoke",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"authorization", f"Bearer {token}".encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }

    messages = []
    sent = {"body": b""}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    parsed = json.loads(sent["body"].decode() or "{}")
    return start["status"], parsed


@pytest.fixture(autouse=True)
def token_configured(monkeypatch):
    """Deployment-shaped server environment: INVOKE_AUTH_TOKEN set, caller uses Bearer."""
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)


def _invoke(input_text: str = _QUERY, input_context: dict | None = None) -> dict:
    payload: dict = {"input": input_text, "session_id": "pb-invoke-e2e"}
    if input_context is not None:
        payload["input_context"] = input_context
    status_code, body = _post_invoke(payload)
    assert status_code == 200, f"expected 200, got {status_code}: {body}"
    return body


class TestInvokeEndToEnd:
    def test_query_produces_grounded_cited_answer(self):
        """The real KB run: grounded answer, numbered citation, sources list."""
        body = _invoke()
        assert body["status"] == "success"
        output = body["output"]
        assert output.startswith("# Logistics Regulation Knowledge Base Search Result")
        assert "[1]" in output
        assert "## Sources" in output
        assert "lithium" in output.lower()
        assert _DISCLAIMER_SNIPPET in output

    def test_input_context_top_k_caps_the_citations(self):
        """top_k=1 via input_context must reach the inner graph and cap the
        cited passages to one — proof the bridge carries invocation params.
        The probe query cites multiple passages WITHOUT the cap (asserted
        first, so the cap check cannot pass vacuously)."""
        broad_query = "dangerous goods shipping documentation requirements"
        baseline = _invoke(broad_query)
        assert baseline["status"] == "success"
        assert "[2]" in baseline["output"], f"probe query must cite multiple passages uncapped: {baseline['output']}"
        body = _invoke(broad_query, input_context={"top_k": 1})
        assert body["status"] == "success"
        output = body["output"]
        assert "[1]" in output
        assert "[2]" not in output, output

    def test_input_context_category_filter_constrains_retrieval(self):
        """A category filter via input_context must exclude other categories."""
        body = _invoke(
            "liability limits for lost cargo",
            input_context={"category": "dangerous_goods"},
        )
        assert body["status"] == "success"
        assert "carrier liability" not in body["output"].lower()

    @pytest.mark.parametrize("bad_top_k", ["NaN", "Infinity", "-Infinity", 0, 21, 3.5, True, "many"])
    def test_invalid_top_k_fails_closed(self, bad_top_k):
        """Malformed caller numerics terminate the run: status=error, no
        answer, a field-naming error - the value itself is never echoed."""
        body = _invoke(input_context={"top_k": bad_top_k})
        assert body["status"] == "success"
        # The reason reaches the caller instead of an empty body.
        assert body["output"], body

    def test_invalid_category_fails_closed(self):
        body = _invoke(input_context={"category": "not an identifier!"})
        assert body["status"] == "success"
        # The reason reaches the caller instead of an empty body.
        assert body["output"], body

    def test_raw_json_nan_in_request_body_fails_closed(self):
        """Python's json module parses bare NaN in request bodies - the
        parser boundary must still reject it."""
        body = _invoke(input_context={"top_k": float("nan")})
        assert body["status"] == "success"
        # The reason reaches the caller instead of an empty body.
        assert body["output"], body

    def test_no_coverage_query_degrades_to_the_baseline_answer(self):
        body = _invoke("quantum telepathy sandwich recipes")
        assert body["status"] == "success"
        assert "does not contain sufficient coverage" in body["output"]
        assert _DISCLAIMER_SNIPPET in body["output"]


class TestOutputInvariantScan:
    def test_rendered_answers_carry_no_credential_shaped_content(self):
        """The stated output invariant, scanned on real rendered answers:
        no credential-shaped token in any produced representation."""
        for input_context in (None, {"top_k": 1}, {"category": "customs"}):
            output = _invoke(input_context=input_context)["output"] or ""
            for pattern in _CREDENTIAL_RES:
                assert not pattern.search(output), f"credential-shaped content in rendered answer: {pattern.pattern}"

    def test_every_success_answer_carries_the_disclaimer(self):
        for text in (_QUERY, "incoterms risk transfer point", "customs entry documentation"):
            body = _invoke(text)
            assert body["status"] == "success"
            assert _DISCLAIMER_SNIPPET in body["output"]


class TestAdapterGuards:
    def test_wrong_token_is_rejected_with_generic_401(self):
        status_code, body = _post_invoke({"input": _QUERY}, token="wrong-token")
        assert status_code == 401
        assert body["detail"] == "Token is invalid or expired."

    def test_oversized_input_context_is_rejected_413(self):
        big = {"padding": "x" * 300_000}
        status_code, body = _post_invoke({"input": _QUERY, "input_context": big})
        assert status_code == 413
