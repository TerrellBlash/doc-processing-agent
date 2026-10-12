"""
TESTS — schema-enforced extraction
====================================
The extract nodes run against a stub model, so these tests need no API
key and no network.

Why these matter: the confidence gate decides what a human reviews by
counting missing required fields. That only works if (1) the schema the
model fills has exactly the fields the gate checks, (2) the model has to
state every field, using null when the document lacks it, and (3) a
failed parse can never look like a complete extraction.

Run with: pytest tests/test_extraction.py -v
"""

import json

import pytest
from langchain_core.messages import AIMessage

import app.nodes as nodes
from app.schemas import EXTRACTION_SCHEMAS, REQUIRED_FIELDS, InvoiceFields

COMPLETE_INVOICE = {
    "vendor_name": "CloudTech Solutions LLC",
    "invoice_number": "INV-2024-0847",
    "invoice_date": "2024-03-15",
    "due_date": "2024-04-15",
    "line_items": [
        {"description": "Cloud Hosting", "quantity": 1, "unit_price": 12000.0, "total": 12000.0}
    ],
    "subtotal": 12000.0,
    "tax": 990.0,
    "total_amount": 12990.0,
    "currency": "USD",
}


class StubModel:
    """Stands in for ChatOpenAI; records the schema it was asked to fill."""

    def __init__(self, result: dict):
        self.result = result
        self.schema = None

    def with_structured_output(self, schema, include_raw=False):
        self.schema = schema
        assert include_raw, "include_raw keeps parse failures from raising"
        return self

    def invoke(self, prompt):
        return self.result


def run_extract(monkeypatch, node, result: dict, doc_type: str = "invoice"):
    stub = StubModel(result)
    monkeypatch.setattr(nodes, "get_llm", lambda config=None: stub)
    state = {
        "document_text": "document text",
        "classification": {"doc_type": doc_type, "confidence": 0.95},
    }
    command = node(state, {})
    return stub, command


def test_schema_fields_are_exactly_the_fields_the_gate_checks():
    for doc_type, schema in EXTRACTION_SCHEMAS.items():
        assert REQUIRED_FIELDS[doc_type] == list(schema.model_fields)


@pytest.mark.parametrize("schema", list(EXTRACTION_SCHEMAS.values()))
def test_every_field_must_be_stated_by_the_model(schema):
    """No defaults: the model must give each field, null when absent, so a
    field can't silently drop out of the output."""
    assert all(field.is_required() for field in schema.model_fields.values())


@pytest.mark.parametrize(
    "node, doc_type",
    [
        (nodes.extract_invoice, "invoice"),
        (nodes.extract_resume, "resume"),
        (nodes.extract_contract, "contract"),
    ],
)
def test_each_extract_node_asks_for_its_own_schema(monkeypatch, node, doc_type):
    stub, _ = run_extract(
        monkeypatch, node, {"parsed": None, "raw": AIMessage(content=""), "parsing_error": None}, doc_type
    )
    assert stub.schema is EXTRACTION_SCHEMAS[doc_type]


def test_complete_extraction_scores_full_confidence(monkeypatch):
    parsed = InvoiceFields.model_validate(COMPLETE_INVOICE)
    _, command = run_extract(
        monkeypatch,
        nodes.extract_invoice,
        {"parsed": parsed, "raw": AIMessage(content="{}"), "parsing_error": None},
    )
    extraction = command.update["extraction"]
    assert command.goto == "validate_extraction"
    assert extraction["confidence"] == 1.0
    assert extraction["missing_fields"] == []
    assert extraction["fields"] == COMPLETE_INVOICE
    json.dumps(extraction["fields"])  # checkpointers must be able to store it


def test_fields_the_model_returns_as_null_count_as_missing(monkeypatch):
    parsed = InvoiceFields.model_validate({**COMPLETE_INVOICE, "due_date": None, "tax": None})
    _, command = run_extract(
        monkeypatch,
        nodes.extract_invoice,
        {"parsed": parsed, "raw": AIMessage(content="{}"), "parsing_error": None},
    )
    extraction = command.update["extraction"]
    assert set(extraction["missing_fields"]) == {"due_date", "tax"}
    assert extraction["confidence"] == round(7 / 9, 4)


def test_failed_parse_goes_to_review_with_zero_confidence(monkeypatch):
    """A parse failure must never pass the gate as a complete extraction."""
    _, command = run_extract(
        monkeypatch,
        nodes.extract_invoice,
        {
            "parsed": None,
            "raw": AIMessage(content="I could not read this document."),
            "parsing_error": ValueError("not valid JSON"),
        },
    )
    extraction = command.update["extraction"]
    assert command.goto == "validate_extraction"
    assert extraction["confidence"] == 0.0
    assert extraction["missing_fields"] == ["all_fields"]
    assert extraction["fields"] == {"raw_response": "I could not read this document."}
