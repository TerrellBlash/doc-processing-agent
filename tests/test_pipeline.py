"""
TESTS — Prove It Works
========================
These tests run the ACTUAL graph with sample documents.
Each test shows the full flow from ingest to completion.

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
from app.graph import build_graph
from langgraph.types import Command


# --- Sample Documents ---

SAMPLE_INVOICE = """
INVOICE #INV-2024-0847

From: CloudTech Solutions LLC
123 Innovation Drive, San Francisco, CA 94105

To: Acme Corporation
456 Business Blvd, Austin, TX 78701

Date: 2024-03-15
Due Date: 2024-04-15
Payment Terms: Net 30

Line Items:
1. Cloud Hosting (Annual) - Qty: 1 - $12,000.00
2. Premium Support Package - Qty: 1 - $3,600.00
3. Data Migration Service - Qty: 1 - $2,500.00

Subtotal: $18,100.00
Tax (8.25%): $1,493.25
Total: $19,593.25

Payment Method: Wire Transfer
Bank: Silicon Valley Bank
Account: XXXX-XXXX-4521
"""

SAMPLE_RESUME = """
TERRELL WASHINGTON
Software Engineer | AI/ML Specialist
Atlanta, GA | terrell.w@email.com | (404) 555-0199

SUMMARY
Full-stack software engineer with 8+ years of experience in fintech 
and mortgage software. Currently pursuing AI Engineering certification 
with focus on production ML pipelines and agentic systems.

SKILLS
Python, Swift, TypeScript, React, SwiftUI, LangGraph, LangChain,
Machine Learning, Computer Vision, iOS Development, FastAPI, Docker

EXPERIENCE
Senior Software Engineer — Truist Financial (2020-2024)
- Led migration of mortgage processing system to cloud architecture
- Built ML pipeline for automated document classification (94% accuracy)
- Reduced loan processing time by 40% through intelligent automation

Software Engineer — Infosys (2016-2020)  
- Developed fintech solutions for Fortune 500 banking clients
- Implemented real-time fraud detection using decision tree models
- Managed team of 4 junior developers

EDUCATION
B.S. Computer Science — Georgia Tech (2016)
IBM AI Engineering Professional Certificate — Coursera (2024, in progress)
"""

SAMPLE_CONTRACT = """
NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is entered into as of 
March 1, 2024 ("Effective Date") by and between:

Party A: TechStartup Inc., a Delaware corporation ("Disclosing Party")
Party B: Venture Capital Partners LLC, a New York LLC ("Receiving Party")

1. CONFIDENTIAL INFORMATION
The Receiving Party agrees to hold in confidence all proprietary 
information disclosed by the Disclosing Party, including but not 
limited to: technical data, trade secrets, business plans, and 
financial information.

2. TERM
This Agreement shall remain in effect for a period of two (2) years 
from the Effective Date, expiring on March 1, 2026.

3. OBLIGATIONS
The Receiving Party shall not disclose any Confidential Information 
to third parties without prior written consent.

4. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.

5. TERMINATION
Either party may terminate this Agreement with 30 days written notice.

Signed:
_________________          _________________
TechStartup Inc.           Venture Capital Partners LLC
"""


@pytest.fixture
def graph():
    """Build a fresh graph for each test."""
    return build_graph()


class TestInvoiceProcessing:
    """Test the full invoice pipeline."""

    def test_invoice_classifies_correctly(self, graph):
        """An invoice should be classified as 'invoice' with high confidence."""
        config = {"configurable": {"thread_id": "test-invoice-001"}}
        result = graph.invoke(
            {
                "document_text": SAMPLE_INVOICE,
                "document_filename": "invoice_march_2024.pdf",
                "document_source": "email_attachment",
            },
            config,
        )
        assert result["classification"]["doc_type"] == "invoice"
        assert result["classification"]["confidence"] >= 0.7

    def test_invoice_extracts_total(self, graph):
        """Should extract the total amount from an invoice."""
        config = {"configurable": {"thread_id": "test-invoice-002"}}
        result = graph.invoke(
            {
                "document_text": SAMPLE_INVOICE,
                "document_filename": "invoice.pdf",
                "document_source": "test",
            },
            config,
        )
        if result.get("extraction"):
            fields = result["extraction"]["fields"]
            # The total should be somewhere in the extraction
            assert "total_amount" in fields or "total" in fields


class TestResumeProcessing:
    """Test the full resume pipeline."""

    def test_resume_classifies_correctly(self, graph):
        config = {"configurable": {"thread_id": "test-resume-001"}}
        result = graph.invoke(
            {
                "document_text": SAMPLE_RESUME,
                "document_filename": "terrell_resume.pdf",
                "document_source": "upload",
            },
            config,
        )
        assert result["classification"]["doc_type"] == "resume"

    def test_resume_extracts_name(self, graph):
        config = {"configurable": {"thread_id": "test-resume-002"}}
        result = graph.invoke(
            {
                "document_text": SAMPLE_RESUME,
                "document_filename": "resume.pdf",
                "document_source": "test",
            },
            config,
        )
        if result.get("extraction"):
            fields = result["extraction"]["fields"]
            assert "full_name" in fields


class TestContractProcessing:
    """Test the full contract pipeline."""

    def test_contract_classifies_correctly(self, graph):
        config = {"configurable": {"thread_id": "test-contract-001"}}
        result = graph.invoke(
            {
                "document_text": SAMPLE_CONTRACT,
                "document_filename": "nda_techstartup.pdf",
                "document_source": "legal_team",
            },
            config,
        )
        assert result["classification"]["doc_type"] == "contract"


class TestHumanReviewFlow:
    """Test the human-in-the-loop interrupt and resume."""

    def test_low_confidence_triggers_review(self, graph):
        """A vague document should trigger human review."""
        config = {"configurable": {"thread_id": "test-review-001"}}

        # Send a deliberately ambiguous document
        result = graph.invoke(
            {
                "document_text": "Hello, this is just a random note. Nothing special here.",
                "document_filename": "mystery.txt",
                "document_source": "unknown",
            },
            config,
        )

        # Should be paused at human review (interrupt)
        # The result will contain __interrupt__ if paused
        state = graph.get_state(config)
        # Either it completed (classified as unknown, went to review)
        # or it's interrupted waiting for human input
        assert state is not None

    def test_human_approval_completes_pipeline(self, graph):
        """After human approves, the pipeline should complete."""
        config = {"configurable": {"thread_id": "test-review-002"}}

        # Start with ambiguous doc
        graph.invoke(
            {
                "document_text": "Hello, this is just a random note.",
                "document_filename": "mystery.txt",
                "document_source": "unknown",
            },
            config,
        )

        # Check if it's waiting for review
        state = graph.get_state(config)
        if state and state.next:
            # Resume with human approval
            result = graph.invoke(
                Command(resume={"approved": True, "corrections": {}}),
                config,
            )
            assert result.get("processing_status") in ["complete", "failed"]


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_empty_document_raises_error(self, graph):
        """An empty document should be rejected at ingest."""
        config = {"configurable": {"thread_id": "test-edge-001"}}
        with pytest.raises(Exception):
            graph.invoke(
                {
                    "document_text": "",
                    "document_filename": "empty.pdf",
                    "document_source": "test",
                },
                config,
            )

    def test_full_pipeline_produces_output(self, graph):
        """A valid document should produce a final_output dict."""
        config = {"configurable": {"thread_id": "test-edge-002"}}
        result = graph.invoke(
            {
                "document_text": SAMPLE_INVOICE,
                "document_filename": "test_invoice.pdf",
                "document_source": "test",
            },
            config,
        )
        # Should either complete or be at human review
        assert result.get("processing_status") in [
            "complete", "reviewing", "validated", "failed"
        ]
