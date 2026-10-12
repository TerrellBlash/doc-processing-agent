"""
Graph state shared by every node.

Each stage writes its own slice:
- ingest: the input document
- classify: document type and classifier confidence
- extract: fields, completeness-based confidence, missing fields
- validate: whether the extraction passes the gate
- human_review: the reviewer's decision

KEY RULE: Store raw data, never formatted prompts.
Each node formats the data how IT needs it.
"""

from typing import TypedDict, Literal


# --- Sub-structures ---

class DocumentClassification(TypedDict):
    """What the classifier figured out about the document."""
    doc_type: Literal["invoice", "resume", "contract", "unknown"]
    confidence: float  # 0.0 to 1.0 — how sure the LLM is
    language: str      # e.g. "english", "spanish"
    summary: str       # one-line description


class ExtractionResult(TypedDict):
    """The structured data pulled from the document."""
    fields: dict           # the actual extracted key-value pairs
    confidence: float      # share of required fields present (0.0 to 1.0)
    missing_fields: list[str]  # fields we expected but couldn't find
    raw_text_used: str     # the chunk of text we extracted from


# --- Main state ---

class DocAgentState(TypedDict):
    # === INPUTS (written once at the start) ===
    document_text: str          # raw text content of the document
    document_filename: str      # original filename for context
    document_source: str        # where it came from (email, upload, etc.)
    document_pages: list[dict] | None  # uploads only: per page, text_layer or vision

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

    # === ENRICHMENT (written by enrich_extraction node) ===
    enrichment: dict | None  # vendor data added via tool call, if any

    # === FINAL OUTPUT ===
    final_output: dict | None       # the cleaned, approved result
    processing_status: Literal[
        "ingested", "classified", "extracted",
        "validated", "reviewing", "complete", "failed"
    ] | None
