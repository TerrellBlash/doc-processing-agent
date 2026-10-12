"""
Graph nodes.

Every node is a Python function that:
  1. Reads the current state
  2. Does its ONE job
  3. Returns what changed

Some nodes also say WHERE TO GO NEXT using Command objects.
That's how the graph branches, e.g. low confidence routes to human review.
"""

import json
import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from app.llm import get_llm
from app.schemas import EXTRACTION_SCHEMAS, REQUIRED_FIELDS
from app.state import DocAgentState, DocumentClassification, ExtractionResult

logger = logging.getLogger(__name__)

class UnprocessableDocumentError(ValueError):
    """The document can't be processed as sent; the client should fix it."""


# ============================================================
# NODE 1: INGEST
# Ingest: validates the input document
# ============================================================
def ingest_document(state: DocAgentState) -> dict:
    """
    First stop: reject empty or unusable input before any LLM call.
    """
    doc_text = state.get("document_text", "")

    if not doc_text or len(doc_text.strip()) < 10:
        raise UnprocessableDocumentError(
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
# Classify: determines document type
# and decides which specialist to send it to
# ============================================================
def classify_document(
    state: DocAgentState, config: RunnableConfig
) -> Command[Literal["extract_invoice", "extract_resume", "extract_contract", "human_review"]]:
    """
    Uses an LLM to figure out: is this an invoice, resume, or contract?
    Then ROUTES to the right extraction specialist.
    
    The LLM's structured classification picks the next node; unknown
    types or confidence below 0.7 route to human review.
    """
    structured_llm = get_llm(config).with_structured_output(DocumentClassification)

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
# NODE 3: EXTRACT (invoice, resume, contract)
# Each node fills its document type's schema from app/schemas.py
# ============================================================
_EXTRACTION_ROLES = {
    "invoice": "You are an expert invoice processor.",
    "resume": "You are an expert resume parser.",
    "contract": "You are an expert contract analyst.",
}


def _extraction_prompt(doc_type: str, document_text: str) -> str:
    return f"""{_EXTRACTION_ROLES[doc_type]}
Extract the fields in the response schema from this {doc_type}.

Rules:
- Use only what the document states. Do not guess or invent values.
- If the document doesn't contain a field, return null for it, or an empty
  list for a list field.
- Write amounts as plain numbers, without currency symbols or thousands
  separators.

{doc_type.capitalize()} text:
{document_text}
"""


def _extract(
    doc_type: str, state: DocAgentState, config: RunnableConfig
) -> Command[Literal["validate_extraction"]]:
    """
    Fill the document type's schema with structured output, then score how
    complete the result is.

    - Parameter doc_type: "invoice", "resume" or "contract".
    - Parameter state: graph state; reads document_text.
    - Parameter config: the run config, passed to get_llm.
    - Returns: a Command to validate_extraction carrying the extraction. If
        the model's output can't be parsed into the schema, confidence is
        0.0 and missing_fields is ["all_fields"], so it can't pass the gate.
    """
    structured_llm = get_llm(config).with_structured_output(
        EXTRACTION_SCHEMAS[doc_type], include_raw=True
    )
    result = structured_llm.invoke(_extraction_prompt(doc_type, state["document_text"]))
    parsed = result.get("parsed")

    if parsed is None:
        logger.warning(
            "Structured extraction failed for %s: %s", doc_type, result.get("parsing_error")
        )
        raw = result.get("raw")
        fields = {"raw_response": raw.text if raw is not None else ""}
        missing = ["all_fields"]
        confidence = 0.0  # zero verified fields were extracted
    else:
        fields = parsed.model_dump()
        confidence, missing = compute_extraction_confidence(doc_type, fields)

    extraction = ExtractionResult(
        fields=fields,
        confidence=confidence,
        missing_fields=missing,
        raw_text_used=state["document_text"][:500],
    )
    return Command(
        update={"extraction": extraction, "processing_status": "extracted"},
        goto="validate_extraction",
    )


def extract_invoice(
    state: DocAgentState, config: RunnableConfig
) -> Command[Literal["validate_extraction"]]:
    """Extract vendor, dates, line items and totals from an invoice."""
    return _extract("invoice", state, config)


def extract_resume(
    state: DocAgentState, config: RunnableConfig
) -> Command[Literal["validate_extraction"]]:
    """Extract contact details, skills, experience and education from a resume."""
    return _extract("resume", state, config)


def extract_contract(
    state: DocAgentState, config: RunnableConfig
) -> Command[Literal["validate_extraction"]]:
    """Extract parties, dates, terms and governing law from a contract."""
    return _extract("contract", state, config)


def compute_extraction_confidence(
    doc_type: str | None, fields: dict
) -> tuple[float, list[str]]:
    """
    Measure how complete an extraction is against the required fields for
    its document type.

    Pure function: no side effects, no network calls. A field counts as
    missing if its key is absent from `fields`, or its value is None, an
    empty string, an empty list, or an empty dict.

    - Parameter doc_type: the classified document type (e.g. "invoice").
        If it is None or not a key in REQUIRED_FIELDS (e.g. "unknown"), there is
        nothing to check completeness against.
    - Parameter fields: the extracted field dict to check.
    - Returns: a tuple of (confidence, missing_fields) where confidence is
        the fraction of required fields present, rounded to 4 decimal
        places, and missing_fields lists the required field names that are
        absent or empty. When doc_type is not in REQUIRED_FIELDS, returns
        (1.0, []).
    """
    required = REQUIRED_FIELDS.get(doc_type) if doc_type else None
    if required is None:
        return 1.0, []

    empty_values = ("", [], {})
    missing = [
        field
        for field in required
        if field not in fields
        or fields[field] is None
        or fields[field] in empty_values
    ]

    confidence = round((len(required) - len(missing)) / len(required), 4)
    return confidence, missing


# ============================================================
# NODE 4: VALIDATE
# Validate: gates extraction on required-field completeness
# ============================================================
def validate_extraction(
    state: DocAgentState,
) -> Command[Literal["enrich_extraction", "human_review"]]:
    """
    Check if the extraction is good enough to store automatically,
    or if a human needs to review it.
    
    This is the CONFIDENCE GATE — the decision point that makes
    this pipeline production-grade instead of a toy demo.
    """
    extraction = state.get("extraction")
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

    # confidence is derived entirely from missing_fields (see
    # compute_extraction_confidence), so a separate `confidence < threshold`
    # check would always fire exactly when this one does — never
    # independently. One real signal, one check; the confidence number
    # still rides along in the message for a human reviewer's benefit.
    if extraction["missing_fields"]:
        issues.append(
            f"Missing fields: {', '.join(extraction['missing_fields'])} "
            f"(confidence: {extraction['confidence']:.2f})"
        )

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
            goto="enrich_extraction",
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
# Human review: interrupt() pauses until a reviewer responds
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
        "document_pages": state.get("document_pages"),
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
# Store: packages the final output
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
        "document_pages": state.get("document_pages"),
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


# ============================================================
# NODE 6: ENRICH EXTRACTION
# Enrich: optional LLM-decided vendor lookup tool call
# ============================================================
def enrich_extraction(
    state: DocAgentState, config: RunnableConfig
) -> Command[Literal["store_results"]]:
    """
    The one agentic node. Binds the vendor_lookup tool to the LLM and
    lets the MODEL decide whether a lookup applies to this document.

    Invoices with a named vendor: the model calls the tool.
    Resumes and most other docs: the model declines, enrichment stays None.

    Either way, this node routes to store_results. It never changes
    which nodes run; its only discretion is whether to call the tool.
    """
    from app.tools import vendor_lookup

    extraction = state.get("extraction") or {}
    fields = extraction.get("fields", {})

    llm_with_tools = get_llm(config).bind_tools([vendor_lookup])

    prompt = (
        "You are enriching extracted document data. If the document names a "
        "vendor or supplier and a vendor lookup would add useful detail, call "
        "the vendor_lookup tool with the vendor name. If there is no vendor "
        "(for example a resume), do not call any tool.\n\n"
        f"Extracted fields:\n{json.dumps(fields, indent=2)}"
    )

    response = llm_with_tools.invoke(prompt)

    enrichment = None
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        call = tool_calls[0]
        vendor_name = call["args"].get("vendor_name", "")
        enrichment = vendor_lookup.invoke({"vendor_name": vendor_name})
        logger.info(f"enrich_extraction: looked up vendor '{vendor_name}'")
    else:
        logger.info("enrich_extraction: no vendor lookup needed")

    return Command(
        update={"enrichment": enrichment},
        goto="store_results",
    )
