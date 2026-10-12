"""
TESTS — compute_extraction_confidence
=======================================
Unit tests for the pure confidence-scoring function. No LLM calls, no
API key, no network — just plain dicts in, a (confidence, missing) tuple
out. These should run in well under a second.

Run with: pytest tests/test_confidence.py -v
"""

from app.nodes import compute_extraction_confidence


class TestInvoiceConfidence:
    """Invoices require all 9 fields from REQUIRED_FIELDS['invoice']."""

    def test_complete_invoice_has_full_confidence(self):
        """Every required field present and non-empty -> confidence 1.0."""
        fields = {
            "vendor_name": "CloudTech Solutions LLC",
            "invoice_number": "INV-2024-0847",
            "invoice_date": "2024-03-15",
            "due_date": "2024-04-15",
            "line_items": [{"description": "Cloud Hosting", "total": 12000.0}],
            "subtotal": 18100.0,
            "tax": 1493.25,
            "total_amount": 19593.25,
            "currency": "USD",
        }
        confidence, missing = compute_extraction_confidence("invoice", fields)
        assert confidence == 1.0
        assert missing == []

    def test_invoice_missing_two_fields(self):
        """Missing total_amount and tax -> 7/9 present, both named as missing.

        This is the case the old hardcoded 0.9-on-success score could never
        catch: valid JSON with incomplete data should NOT score as
        "basically done".
        """
        fields = {
            "vendor_name": "CloudTech Solutions LLC",
            "invoice_number": "INV-2024-0847",
            "invoice_date": "2024-03-15",
            "due_date": "2024-04-15",
            "line_items": [{"description": "Cloud Hosting", "total": 12000.0}],
            "subtotal": 18100.0,
            "currency": "USD",
        }
        confidence, missing = compute_extraction_confidence("invoice", fields)
        assert confidence == round(7 / 9, 4)
        assert set(missing) == {"total_amount", "tax"}
        assert len(missing) == 2


class TestResumeConfidence:
    def test_empty_string_email_counts_as_missing(self):
        """An empty string is present-but-empty, not present-and-valid."""
        fields = {
            "full_name": "Terrell Washington",
            "email": "",
            "phone": "(404) 555-0199",
            "location": "Atlanta, GA",
            "summary": "Full-stack software engineer with 8+ years.",
            "skills": ["Python", "Swift"],
            "experience": [{"company": "Truist", "title": "Senior SWE"}],
            "education": [{"institution": "Georgia Tech", "degree": "B.S."}],
        }
        confidence, missing = compute_extraction_confidence("resume", fields)
        assert missing == ["email"]
        assert confidence == round(7 / 8, 4)


class TestContractConfidence:
    def test_empty_list_key_terms_counts_as_missing(self):
        """An empty list is present-but-empty, not present-and-valid."""
        fields = {
            "contract_type": "NDA",
            "parties": [{"name": "TechStartup Inc.", "role": "Disclosing Party"}],
            "effective_date": "2024-03-01",
            "expiration_date": "2026-03-01",
            "key_terms": [],
            "payment_terms": "N/A",
            "termination_clause": "30 days written notice",
            "governing_law": "Delaware",
        }
        confidence, missing = compute_extraction_confidence("contract", fields)
        assert missing == ["key_terms"]
        assert confidence == round(7 / 8, 4)


class TestUnknownDocType:
    def test_unknown_doc_type_has_full_confidence_and_no_missing(self):
        """Nothing to check completeness against -> (1.0, [])."""
        confidence, missing = compute_extraction_confidence(
            "unknown", {"raw_response": "some unparseable text"}
        )
        assert confidence == 1.0
        assert missing == []
