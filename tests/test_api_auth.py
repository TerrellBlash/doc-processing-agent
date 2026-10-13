"""
TESTS — API key check
=======================
The graph is stubbed, so these run offline.

Why these matter: every accepted request spends OpenAI credits on the
owner's key. A deployed service with a public URL must turn strangers away
before it reads their upload or calls a model, while Azure's health probe
and the API docs stay reachable without a key.

Run with: pytest tests/test_api_auth.py -v
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main

KEY = "test-key-123"
DOCUMENT = {"document_text": "INVOICE INV-1001\nVendor: Acme Corp\nTotal due: $1,250.00"}


class StubGraph:
    def __init__(self):
        self.calls = 0

    def invoke(self, state, config):
        self.calls += 1
        return {"processing_status": "complete", "final_output": {"ok": True}}


@pytest.fixture
def graph(monkeypatch):
    stub = StubGraph()
    monkeypatch.setattr(main, "graph", stub)
    return stub


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def key_required(monkeypatch):
    monkeypatch.setenv("DOC_AGENT_API_KEY", KEY)


def test_request_without_a_key_is_refused_before_any_model_call(client, graph, key_required):
    response = client.post("/process", json=DOCUMENT)
    assert response.status_code == 401
    assert graph.calls == 0


def test_wrong_key_is_refused(client, graph, key_required):
    response = client.post("/process", json=DOCUMENT, headers={"X-API-Key": "guess"})
    assert response.status_code == 401
    assert graph.calls == 0


def test_upload_without_a_key_is_refused_before_it_is_read(client, graph, key_required):
    files = {"file": ("invoice.pdf", b"%PDF-1.4 whatever", "application/pdf")}
    response = client.post("/process/file", files=files)
    assert response.status_code == 401


def test_right_key_is_accepted(client, graph, key_required):
    response = client.post("/process", json=DOCUMENT, headers={"X-API-Key": KEY})
    assert response.status_code == 200
    assert graph.calls == 1


@pytest.mark.parametrize("path", ["/health", "/docs", "/openapi.json"])
def test_health_probe_and_docs_need_no_key(client, key_required, path):
    assert client.get(path).status_code == 200


def test_no_key_configured_leaves_local_development_open(client, graph, monkeypatch):
    monkeypatch.delenv("DOC_AGENT_API_KEY", raising=False)
    assert client.post("/process", json=DOCUMENT).status_code == 200
