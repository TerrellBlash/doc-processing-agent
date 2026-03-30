"""
STATE DESIGN — The Shared Notebook
===================================
Every node in our graph reads from and writes to this state.

Think of it like a clipboard being passed between specialists:
- The receptionist writes down what document came in
- The classifier writes what type it is
- The extractor writes what data it found
- The validator writes whether the extraction looks right
- The reviewer (human) writes their approval

KEY RULE: Store raw data, never formatted prompts.
Each node formats the data how IT needs it.
"""

from typing import TypedDict, Literal


# --- Sub-structures (pieces of the notebook) ---

class DocumentClassification(TypedDict):
    """What the classifier figured out about the document."""
    doc_type: Literal["invoice", "resume", "contract", "unknown"]
    confidence: float  # 0.0 to 1.0 — how sure the LLM is
    language: str      # e.g. "english", "spanish"
    summary: str       # one-line description


class ExtractionResult(TypedDict):
    """The structured data pulled from the document."""
    fields: dict           # the actual extracted key-value pairs
    confidence: float      # overall extraction confidence
    missing_fields: list[str]  # fields we expected but couldn't find
    raw_text_used: str     # the chunk of text we extracted from


# --- The Main State (the full notebook) ---

class DocAgentState(TypedDict):
    # === INPUTS (written once at the start) ===
    document_text: str          # raw text content of the document
    document_filename: str      # original filename for context
    document_source: str        # where it came from (email, upload, etc.)

    # === CLASSIFICATION (written by classify node) ===
    classification: DocumentClassification | None

    # === EXTRACTION (written by extract nodes) ===
    extraction: ExtractionResult | None

    # === VALIDATION (written by validate node) ===
    validation_passed: bool | None
    validation_issues: list[str] | None

    # === HUMAN REVIEW (written by human review node) ===
    human_approved: bool | None
    human_corrections: dict | None  # any edits the human made

    # === FINAL OUTPUT ===
    final_output: dict | None       # the cleaned, approved result
    processing_status: Literal[
        "ingested", "classified", "extracted",
        "validated", "reviewing", "complete", "failed"
    ] | None
