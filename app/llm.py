"""
Chat model construction.

Every node gets its model from get_llm(), so the model can be chosen per
run (graph config) or per deployment (DOC_AGENT_MODEL) without code
changes. The evaluation harness relies on this to compare models on the
same dataset.
"""

import os
from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_REASONING_EFFORT = "none"

# Models from before reasoning controls sample with a temperature, and 0
# keeps extraction steady. Current reasoning models reject temperature=0
# with a 400 error and take reasoning_effort instead. The default effort is
# "none": extraction is reading, not multi-step reasoning, and the eval
# measures what that costs in accuracy.
_TEMPERATURE_MODELS = ("gpt-4o", "gpt-4.1")


def resolve_model(config: RunnableConfig | None = None) -> str:
    """
    Pick the model for this run.

    - Parameter config: the graph's run config. A `model` key under
        `configurable` wins; otherwise DOC_AGENT_MODEL, then DEFAULT_MODEL.
    - Returns: the OpenAI model name.
    """
    configurable = (config or {}).get("configurable") or {}
    return configurable.get("model") or os.environ.get("DOC_AGENT_MODEL") or DEFAULT_MODEL


@lru_cache(maxsize=8)
def _build(model: str, reasoning_effort: str) -> ChatOpenAI:
    if model.startswith(_TEMPERATURE_MODELS):
        return ChatOpenAI(model=model, temperature=0)
    # Chat Completions rejects function tools combined with a reasoning
    # effort on current models, and enrich_extraction binds a tool. Any
    # effort other than "none" therefore goes through the Responses API.
    return ChatOpenAI(
        model=model,
        reasoning_effort=reasoning_effort,
        use_responses_api=reasoning_effort != "none",
    )


def get_llm(config: RunnableConfig | None = None) -> ChatOpenAI:
    """
    Return a chat client for the run's model, built once per model.

    - Parameter config: the graph's run config, passed through from the node.
    - Returns: a ChatOpenAI client. Reasoning models get
        DOC_AGENT_REASONING_EFFORT (default "none"); older models get
        temperature 0.
    """
    effort = os.environ.get("DOC_AGENT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    return _build(resolve_model(config), effort)
