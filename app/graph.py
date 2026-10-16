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

import logging
import os

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
    enrich_extraction,
)

logger = logging.getLogger(__name__)


def build_checkpointer():
    """
    Build the checkpointer the graph will use to persist state.

    Why this matters: a paused `human_review` interrupt only lives inside
    whichever checkpointer backs the graph. The API server runs multiple
    uvicorn workers (separate processes, separate memory), and a container
    restart wipes process memory entirely. An in-memory checkpointer means
    a review submitted to a different worker than the one that paused it
    404s, and any restart silently drops every pending review.

    If DATABASE_URL is set, this builds a connection pool and a
    PostgresSaver backed by it, so paused state lives in the database
    instead of a single process's RAM — any worker, and a restart, can
    pick it back up. If it's not set, this falls back to MemorySaver,
    which is fine for local development and tests, but NOT fine for a
    multi-worker or restart-prone deployment.

    This function always returns a usable checkpointer; it never raises.

    - Returns: a `PostgresSaver` when `DATABASE_URL` is configured,
        otherwise a `MemorySaver`.
    """
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        logger.warning(
            "No DATABASE_URL configured — falling back to in-memory "
            "checkpointing. Paused reviews will NOT survive a restart or "
            "a different worker process. This is only acceptable for "
            "local development or tests."
        )
        return MemorySaver()

    # Imported lazily so environments without DATABASE_URL (and without
    # these optional dependencies installed) never need them.
    from psycopg_pool import ConnectionPool
    from psycopg import Connection
    from psycopg.rows import DictRow, dict_row
    from langgraph.checkpoint.postgres import PostgresSaver

    # check= tests each connection before handing it out and replaces a
    # dead one. Serverless Postgres (Neon) closes every connection when it
    # suspends after a few idle minutes, and networks drop idle ones too;
    # without the check, the next request after a quiet spell fails.
    pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
        conninfo=database_url,
        kwargs={"autocommit": True, "row_factory": dict_row},
        check=ConnectionPool[Connection[DictRow]].check_connection,
    )
    checkpointer = PostgresSaver(pool)

    # .setup() is idempotent ONCE the schema exists — but on a freshly
    # provisioned, not-yet-migrated database it is not safe to call
    # concurrently. It runs with autocommit (each statement commits as it
    # runs, not as one transaction), so two workers can both read "no
    # migrations applied yet" before either one's insert commits, and the
    # second insert then collides on a primary key. The Dockerfile starts
    # 4 uvicorn workers, each calling build_checkpointer() independently on
    # startup, so this race is real on first deploy against an empty
    # database. A Postgres advisory lock serializes it: whichever worker
    # gets here first runs the migration, the rest block until it's done,
    # then find there's nothing left to do.
    _SETUP_LOCK_KEY = 0x646F6361  # arbitrary fixed key, spells "doca"
    with pool.connection() as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_SETUP_LOCK_KEY,))
        try:
            checkpointer.setup()
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_SETUP_LOCK_KEY,))

    # Stashed under a prefixed name (not a PostgresSaver attribute) so
    # close_checkpointer() can shut the pool down on app shutdown. Without
    # this, every build_graph() call — including the second graph each
    # restart test builds — leaks a live connection pool.
    setattr(checkpointer, "_doc_agent_pool", pool)

    logger.info("Persistent Postgres checkpointing is active (DATABASE_URL configured).")

    return checkpointer


def close_checkpointer(checkpointer):
    """
    Release whatever resources `checkpointer` is holding, if any.

    A MemorySaver holds nothing and this is a no-op. A PostgresSaver built
    by `build_checkpointer()` carries a live connection pool stashed at
    `_doc_agent_pool`; this closes it. Call this on app shutdown (see
    app/main.py's lifespan) and in any test that builds its own graph
    directly, so pools don't accumulate for the life of a long-running
    test session or a dev server that restarts often.
    """
    pool = getattr(checkpointer, "_doc_agent_pool", None)
    if pool is not None:
        pool.close()


def build_graph(checkpointer=None):
    """
    Build and compile the document processing graph.

    - Parameter checkpointer: the checkpointer to compile the graph with.
        When omitted (None), `build_checkpointer()` is called to pick one
        based on whether DATABASE_URL is configured. Tests that want to
        simulate a worker restart build two independent graphs this way —
        each call to `build_checkpointer()` (and, for Postgres, each new
        connection pool) stands in for a separate worker process, while
        the underlying Postgres database is what actually persists state
        across them.
    - Returns: the compiled graph ready to invoke.
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
    workflow.add_node("enrich_extraction", enrich_extraction)

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
    # The checkpointer enables interrupt() to pause and resume. When the
    # caller doesn't supply one, build_checkpointer() picks Postgres
    # (DATABASE_URL set) or MemorySaver (local/test fallback) for us.
    if checkpointer is None:
        checkpointer = build_checkpointer()
    graph = workflow.compile(checkpointer=checkpointer)

    return graph


# Pre-built graph instance for the API server
graph = build_graph()
