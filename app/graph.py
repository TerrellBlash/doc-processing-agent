"""
GRAPH WIRING — Connecting the Stops
=====================================
This is where we tell LangGraph:
  "Start here, then go here, end here."

Most of the routing happens INSIDE the nodes via Command objects.
Here we only define:
  1. The entry point (START → ingest)
  2. The mandatory sequential edges (ingest → classify)
  3. The exit point (store_results → END)

The nodes themselves handle all the branching decisions.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from app.state import DocAgentState
from app.nodes import (
    ingest_document,
    classify_document,
    extract_invoice,
    extract_resume,
    extract_contract,
    validate_extraction,
    human_review,
    store_results,
)


def build_graph():
    """
    Build and compile the document processing graph.
    
    Returns the compiled graph ready to invoke.
    """
    workflow = StateGraph(DocAgentState)

    # --- ADD NODES ---
    # Each node is a function that takes state and returns updates.
    workflow.add_node("ingest", ingest_document)
    workflow.add_node("classify", classify_document)

    # Extraction nodes — each specialist gets a retry policy
    # because LLM calls can be flaky (rate limits, timeouts)
    workflow.add_node(
        "extract_invoice",
        extract_invoice,
        retry_policy=RetryPolicy(max_attempts=3),
    )
    workflow.add_node(
        "extract_resume",
        extract_resume,
        retry_policy=RetryPolicy(max_attempts=3),
    )
    workflow.add_node(
        "extract_contract",
        extract_contract,
        retry_policy=RetryPolicy(max_attempts=3),
    )

    workflow.add_node("validate_extraction", validate_extraction)
    workflow.add_node("human_review", human_review)
    workflow.add_node("store_results", store_results)

    # --- ADD EDGES ---
    # Only the "always go here next" connections.
    # Branching is handled by Command objects inside nodes.
    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "classify")
    # classify → extract_* (handled by Command in classify_document)
    # extract_* → validate (handled by Command in extract functions)
    # validate → store OR human_review (handled by Command in validate)
    # human_review → store (handled by Command in human_review)
    workflow.add_edge("store_results", END)

    # --- COMPILE ---
    # MemorySaver enables interrupt() to pause and resume.
    # In production on Azure, you'd swap this for a Redis or
    # PostgreSQL checkpointer so state survives container restarts.
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    return graph


# Pre-built graph instance for the API server
graph = build_graph()
