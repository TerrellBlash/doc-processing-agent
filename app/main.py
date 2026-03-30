"""
API SERVER — The Front Door
==============================
This is what Azure Container Apps actually runs.
It's a FastAPI server with three endpoints:

  POST /process     → Start processing a document
  POST /review      → Submit a human review decision  
  GET  /status/{id} → Check how a document is doing

Your friend's Twilio setup would call these endpoints.
A web dashboard would call these endpoints.
Your DiceVault iOS app would call these endpoints.
Same pattern every time.
"""

import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.types import Command

from app.graph import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Request/Response Models ---

class ProcessRequest(BaseModel):
    """What the client sends to start processing a document."""
    document_text: str
    document_filename: str = "unknown.pdf"
    document_source: str = "api_upload"


class ReviewRequest(BaseModel):
    """What a human reviewer sends to approve/reject."""
    thread_id: str
    approved: bool
    corrections: dict | None = None


class StatusResponse(BaseModel):
    """What we send back about the current state of processing."""
    thread_id: str
    status: str
    classification: dict | None = None
    extraction_preview: dict | None = None
    needs_review: bool = False
    review_context: dict | None = None
    final_output: dict | None = None


# --- App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Document Processing Agent starting up")
    yield
    logger.info("Document Processing Agent shutting down")


app = FastAPI(
    title="Intelligent Document Processing Agent",
    description=(
        "LangGraph-powered pipeline that classifies, extracts, "
        "validates, and routes documents with human-in-the-loop review."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- Endpoints ---

@app.post("/process", response_model=StatusResponse)
async def process_document(request: ProcessRequest):
    """
    Start processing a new document.
    
    The graph will run until it either:
    - Completes successfully (high confidence extraction)
    - Pauses at human_review (needs approval via /review)
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "document_text": request.document_text,
        "document_filename": request.document_filename,
        "document_source": request.document_source,
    }

    logger.info(f"Processing document: {request.document_filename} (thread: {thread_id})")

    try:
        result = graph.invoke(initial_state, config)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return _build_status_response(thread_id, result)


@app.post("/review", response_model=StatusResponse)
async def submit_review(request: ReviewRequest):
    """
    Submit a human review decision for a paused document.
    
    This is the other half of interrupt() — the graph has been
    sleeping since it hit the human_review node. This wakes it up.
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    # Build the Command that resumes the graph
    human_response = Command(
        resume={
            "approved": request.approved,
            "corrections": request.corrections or {},
        }
    )

    logger.info(
        f"Resuming review for thread {request.thread_id} "
        f"(approved: {request.approved})"
    )

    try:
        result = graph.invoke(human_response, config)
    except Exception as e:
        logger.error(f"Review processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return _build_status_response(request.thread_id, result)


@app.get("/status/{thread_id}", response_model=StatusResponse)
async def get_status(thread_id: str):
    """
    Check the current status of a document being processed.
    
    Reads from the checkpointer — the saved state from the
    last time the graph ran for this thread.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
        if not state or not state.values:
            raise HTTPException(
                status_code=404,
                detail=f"No document found for thread {thread_id}",
            )
        return _build_status_response(thread_id, state.values)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Azure Container Apps uses this to know the service is alive."""
    return {"status": "healthy", "service": "doc-processing-agent"}


# --- Helper ---

def _build_status_response(thread_id: str, state: dict) -> StatusResponse:
    """Build a clean status response from raw graph state."""
    status = state.get("processing_status", "unknown")
    classification = state.get("classification")
    extraction = state.get("extraction")

    # Check if the graph is paused waiting for human review
    needs_review = status == "reviewing"
    review_context = None
    if needs_review:
        review_context = {
            "issues": state.get("validation_issues", []),
            "classification": classification,
            "extraction_preview": (
                extraction.get("fields", {}) if extraction else None
            ),
        }

    return StatusResponse(
        thread_id=thread_id,
        status=status,
        classification=classification,
        extraction_preview=(
            extraction.get("fields", {}) if extraction else None
        ),
        needs_review=needs_review,
        review_context=review_context,
        final_output=state.get("final_output"),
    )
