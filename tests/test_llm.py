"""
TESTS — get_llm model selection
================================
No network calls: building a ChatOpenAI client doesn't contact the API.

Why these matter: current OpenAI reasoning models reject temperature=0
with a 400 error, so sending the old setting to them would fail every
call in the pipeline. And the evaluation harness compares models by
passing one per run, which only works if that choice actually reaches
the client.

Run with: pytest tests/test_llm.py -v
"""

import pytest

from app.llm import DEFAULT_MODEL, get_llm


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """A dummy key so the client can be built offline; no model override."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.delenv("DOC_AGENT_MODEL", raising=False)
    monkeypatch.delenv("DOC_AGENT_REASONING_EFFORT", raising=False)


def test_default_model_is_used_when_nothing_is_configured():
    assert get_llm().model_name == DEFAULT_MODEL


def test_reasoning_model_gets_reasoning_effort_and_no_temperature():
    """gpt-6 models only accept the default temperature."""
    llm = get_llm({"configurable": {"model": "gpt-6-luna"}})
    assert llm.model_name == "gpt-6-luna"
    assert llm.temperature is None
    assert llm.reasoning_effort == "none"
    assert not llm.use_responses_api


def test_older_model_keeps_temperature_zero():
    """gpt-4o-era models predate reasoning effort; temperature 0 keeps them steady."""
    llm = get_llm({"configurable": {"model": "gpt-4o-mini"}})
    assert llm.temperature == 0
    assert llm.reasoning_effort is None


def test_env_var_sets_the_deployment_model(monkeypatch):
    monkeypatch.setenv("DOC_AGENT_MODEL", "gpt-5.4-mini")
    assert get_llm().model_name == "gpt-5.4-mini"


def test_per_run_config_overrides_the_env_var(monkeypatch):
    """The eval harness picks the model per run; that must win over the deployment default."""
    monkeypatch.setenv("DOC_AGENT_MODEL", "gpt-5.4-mini")
    assert get_llm({"configurable": {"model": "gpt-4o-mini"}}).model_name == "gpt-4o-mini"


def test_reasoning_effort_above_none_uses_the_responses_api(monkeypatch):
    """Chat Completions returns a 400 for tools plus a reasoning effort on
    gpt-6 models, which would break the enrich_extraction tool call."""
    monkeypatch.setenv("DOC_AGENT_REASONING_EFFORT", "medium")
    llm = get_llm({"configurable": {"model": "gpt-6-luna"}})
    assert llm.reasoning_effort == "medium"
    assert llm.use_responses_api


def test_graph_passes_the_run_model_to_its_nodes(monkeypatch):
    """If the run's model never reached the nodes, an eval "comparing"
    models would silently run the same model every time."""
    from langgraph.checkpoint.memory import MemorySaver

    import app.nodes as nodes
    from app.graph import build_graph
    from app.llm import resolve_model

    seen = []

    class StopBeforeNetwork(Exception):
        pass

    def recording_get_llm(config=None):
        seen.append(resolve_model(config))
        raise StopBeforeNetwork

    monkeypatch.setattr(nodes, "get_llm", recording_get_llm)
    graph = build_graph(MemorySaver())
    with pytest.raises(StopBeforeNetwork):
        graph.invoke(
            {
                "document_text": "INVOICE #1 from Acme Corp. Total due: $5.00",
                "document_filename": "invoice.txt",
                "document_source": "test",
            },
            {"configurable": {"thread_id": "model-wiring", "model": "gpt-6-luna"}},
        )
    assert seen == ["gpt-6-luna"]
