"""
NODES — Each Stop on the Adventure
====================================
Every node is a Python function that:
  1. Reads the current state (the notebook)
  2. Does its ONE job
  3. Returns what changed

Some nodes also say WHERE TO GO NEXT using Command objects.
That's how the graph makes decisions — "if urgent, go to human review."
"""

import json
import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt

from app.state import DocAgentState, DocumentClassification, ExtractionResult

logger = logging.getLogger(__name__)

# --- LLM Setup ---
# You can swap this for Gemini, Claude, etc.
# Just change the import and model name.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# NODE 1: INGEST
# "The receptionist" — reads the document, confirms it's valid
# ============================================================
def ingest_document(state: DocAgentState) -> dict:
    """
    First stop: make sure we actually have a document to work with.
    
    Like a receptionist checking that a patient actually filled out
    their paperwork before sending them to the doctor.
    """
    doc_text = state.get("document_text", "")

    if not doc_text or len(doc_text.strip()) < 10:
        raise ValueError(
            "Document is empty or too short to process. "
            "Please provide actual document content."
        )

    logger.info(
        f"Ingested document: {state.get('document_filename', 'unknown')} "
        f"({len(doc_text)} chars)"
    )

    return {"processing_status": "ingested"}


# ============================================================
# NODE 2: CLASSIFY
# "The triage nurse" — figures out what type of document this is
# and decides which specialist to send it to
# ============================================================
def classify_document(
    state: DocAgentState,
) -> Command[Literal["extract_invoice", "extract_resume", "extract_contract", "human_review"]]:
    """
    Uses an LLM to figure out: is this an invoice, resume, or contract?
    Then ROUTES to the right extraction specialist.
    
    This is where the 'choose your own adventure' happens —
    the LLM reads the document and picks the next path.
    """
    structured_llm = llm.with_structured_output(DocumentClassification)

    # Format the prompt ON DEMAND (not stored in state)
    prompt = f"""Analyze this document and classify it.

Document filename: {state.get('document_filename', 'unknown')}
Source: {state.get('document_source', 'unknown')}

Document content (first 3000 chars):
{state['document_text'][:3000]}

Classify as one of: invoice, resume, contract, unknown.
Provide your confidence (0.0 to 1.0), the language, and a one-line summary.
"""

    classification = structured_llm.invoke(prompt)
    doc_type = classification["doc_type"]
    confidence = classification["confidence"]

    logger.info(f"Classified as: {doc_type} (confidence: {confidence:.2f})")

    # --- ROUTING DECISION ---
    # If we're not confident enough OR it's unknown, ask a human
    if doc_type == "unknown" or confidence < 0.7:
        return Command(
            update={
                "classification": classification,
                "processing_status": "classified",
            },
            goto="human_review",
        )

    # Otherwise, route to the right specialist
    route_map = {
        "invoice": "extract_invoice",
        "resume": "extract_resume",
        "contract": "extract_contract",
    }

    return Command(
        update={
            "classification": classification,
            "processing_status": "classified",
        },
        goto=route_map[doc_type],
    )


# ============================================================
# NODE 3a: EXTRACT INVOICE
# "The accountant" — pulls out line items, totals, dates
# ============================================================
def extract_invoice(
    state: DocAgentState,
) -> Command[Literal["validate_extraction"]]:
    """Extract structured data from an invoice document."""
    prompt = f"""You are an expert invoice processor.
Extract the following fields from this invoice as JSON:

Required fields:
- vendor_name: string
- invoice_number: string  
- invoice_date: string (ISO format)
- due_date: string (ISO format) 
- line_items: list of {{ description, quantity, unit_price, total }}
- subtotal: number
- tax: number
- total_amount: number
- currency: string

Invoice text:
{state['document_text']}

Return ONLY valid JSON, no other text.
"""
    response = llm.invoke(prompt)
    return _parse_extraction(response.content, state, "validate_extraction")


# ============================================================
# NODE 3b: EXTRACT RESUME
# "The recruiter" — pulls out skills, experience, education
# ============================================================
def extract_resume(
    state: DocAgentState,
) -> Command[Literal["validate_extraction"]]:
    """Extract structured data from a resume."""
    prompt = f"""You are an expert resume parser.
Extract the following fields from this resume as JSON:

Required fields:
- full_name: string
- email: string
- phone: string
- location: string
- summary: string (professional summary)
- skills: list of strings
- experience: list of {{ company, title, start_date, end_date, description }}
- education: list of {{ institution, degree, field, graduation_date }}

Resume text:
{state['document_text']}

Return ONLY valid JSON, no other text.
"""
    response = llm.invoke(prompt)
    return _parse_extraction(response.content, state, "validate_extraction")


# ============================================================
# NODE 3c: EXTRACT CONTRACT
# "The lawyer" — pulls out terms, parties, dates, obligations
# ============================================================
def extract_contract(
    state: DocAgentState,
) -> Command[Literal["validate_extraction"]]:
    """Extract structured data from a contract."""
    prompt = f"""You are an expert contract analyst.
Extract the following fields from this contract as JSON:

Required fields:
- contract_type: string (e.g. "NDA", "Service Agreement", "Employment")
- parties: list of {{ name, role }} 
- effective_date: string (ISO format)
- expiration_date: string (ISO format)
- key_terms: list of strings (main obligations/terms)
- payment_terms: string
- termination_clause: string
- governing_law: string (jurisdiction)

Contract text:
{state['document_text']}

Return ONLY valid JSON, no other text.
"""
    response = llm.invoke(prompt)
    return _parse_extraction(response.content, state, "validate_extraction")


def _parse_extraction(
    raw_response: str, state: DocAgentState, next_node: str
) -> Command:
    """
    Helper: parse LLM extraction response into structured result.
    Handles the messy reality of LLM outputs — sometimes they add
    markdown backticks, sometimes they hallucinate extra text.
    """
    # Clean up common LLM response quirks
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        fields = json.loads(cleaned)
        missing = []
        confidence = 0.9  # base confidence for successful parse
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM extraction as JSON")
        fields = {"raw_response": raw_response}
        missing = ["all_fields"]
        confidence = 0.3

    extraction = ExtractionResult(
        fields=fields,
        confidence=confidence,
        missing_fields=missing,
        raw_text_used=state["document_text"][:500],
    )

    return Command(
        update={"extraction": extraction, "processing_status": "extracted"},
        goto=next_node,
    )


# ============================================================
# NODE 4: VALIDATE
# "The quality checker" — makes sure the extraction looks right
# ============================================================
def validate_extraction(
    state: DocAgentState,
) -> Command[Literal["store_results", "human_review"]]:
    """
    Check if the extraction is good enough to store automatically,
    or if a human needs to review it.
    
    This is the CONFIDENCE GATE — the decision point that makes
    this pipeline production-grade instead of a toy demo.
    """
    extraction = state.get("extraction")
    classification = state.get("classification")
    issues = []

    if not extraction:
        issues.append("No extraction result found")
        return Command(
            update={
                "validation_passed": False,
                "validation_issues": issues,
                "processing_status": "reviewing",
            },
            goto="human_review",
        )

    # Check extraction confidence
    if extraction["confidence"] < 0.85:
        issues.append(
            f"Low extraction confidence: {extraction['confidence']:.2f}"
        )

    # Check for missing required fields
    if extraction["missing_fields"]:
        issues.append(
            f"Missing fields: {', '.join(extraction['missing_fields'])}"
        )

    # Cross-check: does the extraction match the classification?
    if classification and extraction["fields"]:
        doc_type = classification["doc_type"]
        fields = extraction["fields"]

        if doc_type == "invoice" and "total_amount" not in fields:
            issues.append("Invoice missing total_amount")
        elif doc_type == "resume" and "full_name" not in fields:
            issues.append("Resume missing full_name")
        elif doc_type == "contract" and "parties" not in fields:
            issues.append("Contract missing parties")

    # --- THE CONFIDENCE GATE ---
    passed = len(issues) == 0

    if passed:
        logger.info("Validation passed — auto-storing results")
        return Command(
            update={
                "validation_passed": True,
                "validation_issues": [],
                "processing_status": "validated",
            },
            goto="store_results",
        )
    else:
        logger.info(f"Validation failed ({len(issues)} issues) — sending to human review")
        return Command(
            update={
                "validation_passed": False,
                "validation_issues": issues,
                "processing_status": "reviewing",
            },
            goto="human_review",
        )


# ============================================================
# NODE 5: HUMAN REVIEW
# "The supervisor" — pauses everything and waits for a human
# ============================================================
def human_review(
    state: DocAgentState,
) -> Command[Literal["store_results"]]:
    """
    This is where interrupt() does its magic.
    
    The graph STOPS here. Saves everything. Goes to sleep.
    Could be 5 seconds or 5 days — doesn't matter.
    When a human responds, it wakes up and continues
    EXACTLY where it left off.
    
    interrupt() MUST be the first thing in the function.
    Any code before it will re-run when the graph resumes.
    """
    # Gather context for the human reviewer
    classification = state.get("classification", {})
    extraction = state.get("extraction", {})

    # === PAUSE HERE ===
    human_decision = interrupt({
        "message": "Document needs human review",
        "document_filename": state.get("document_filename", "unknown"),
        "classification": classification,
        "extraction_preview": {
            k: v for k, v in extraction.get("fields", {}).items()
            if not isinstance(v, list)  # show simple fields only
        },
        "issues": state.get("validation_issues", []),
        "action_needed": "Please review and approve/correct the extraction",
    })
    # === RESUMES HERE when human responds ===

    approved = human_decision.get("approved", False)
    corrections = human_decision.get("corrections", {})

    if approved:
        # If human provided corrections, merge them
        if corrections and state.get("extraction"):
            updated_fields = {**state["extraction"]["fields"], **corrections}
            updated_extraction = {**state["extraction"], "fields": updated_fields}
            return Command(
                update={
                    "human_approved": True,
                    "human_corrections": corrections,
                    "extraction": updated_extraction,
                    "processing_status": "validated",
                },
                goto="store_results",
            )
        else:
            return Command(
                update={
                    "human_approved": True,
                    "processing_status": "validated",
                },
                goto="store_results",
            )
    else:
        # Human rejected — mark as failed
        return Command(
            update={
                "human_approved": False,
                "processing_status": "failed",
            },
            goto="store_results",
        )


# ============================================================
# NODE 6: STORE RESULTS
# "The filing clerk" — packages everything up as the final output
# ============================================================
def store_results(state: DocAgentState) -> dict:
    """
    Final stop: package the extraction into a clean output.
    
    In production, this would write to a database.
    For now, we structure it as JSON ready to be stored anywhere.
    """
    extraction = state.get("extraction", {})
    classification = state.get("classification", {})

    final_output = {
        "document_filename": state.get("document_filename", "unknown"),
        "document_type": classification.get("doc_type", "unknown"),
        "classification_confidence": classification.get("confidence", 0),
        "extracted_data": extraction.get("fields", {}),
        "extraction_confidence": extraction.get("confidence", 0),
        "human_reviewed": state.get("human_approved") is not None,
        "human_approved": state.get("human_approved"),
        "processing_status": (
            "complete" if state.get("processing_status") != "failed" else "failed"
        ),
    }

    logger.info(
        f"Stored results for {final_output['document_filename']} "
        f"(type: {final_output['document_type']}, "
        f"status: {final_output['processing_status']})"
    )

    return {
        "final_output": final_output,
        "processing_status": final_output["processing_status"],
    }
