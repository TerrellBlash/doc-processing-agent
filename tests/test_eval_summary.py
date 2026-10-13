"""
TESTS — eval summary metrics and the per-document runner
==========================================================
Hand-built records and stub graphs, no network.

Why these matter: the routing numbers are the headline claim of a
human-in-the-loop pipeline. A document stored without review that has a
wrong field is a silent error, the costliest kind, and the summary must
count it as one instead of averaging it away. The runner must not hand
the model the answer, and a document that crashes must count against the
model, not drop out of the averages.

Run with: pytest tests/test_eval_summary.py -v
"""

from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

import evals.run_eval as run_eval
from evals.pricing import cost_usd
from evals.run_eval import markdown_report, run_document, summarize


def record(doc_id, expected_route, route, outcomes, doc_type="invoice", classified_as=None,
           absent_field=None, fmt="text", seconds=1.0, cost=0.001):
    return {
        "id": doc_id, "doc_type": doc_type, "classified_as": classified_as or doc_type, "format": fmt,
        "expected_route": expected_route, "route": route, "outcomes": outcomes,
        "absent_field": absent_field, "seconds": seconds, "cost_usd": cost,
        "input_tokens": 1000, "output_tokens": 100, "error": None,
    }


RECORDS = [
    record("a", "auto", "auto", {"vendor_name": "correct", "total_amount": "correct"}),
    record("b", "auto", "auto", {"vendor_name": "correct", "total_amount": "wrong"}, fmt="pdf_scan"),
    record("c", "auto", "review", {"vendor_name": "correct", "total_amount": "missed"}),
    record("d", "review", "review", {"vendor_name": "correct", "due_date": "correct"}, absent_field="due_date"),
    record("e", "review", "auto", {"vendor_name": "correct", "due_date": "filled_absent"}, absent_field="due_date"),
    record("f", "review", "review", {}, doc_type="unknown"),
]


def test_a_wrong_field_in_an_auto_stored_document_is_a_silent_error():
    summary = summarize(RECORDS)
    # Auto-stored: a (clean), b (wrong total), e (should have gone to review).
    assert summary["auto_stored"] == 3
    assert summary["silent_errors"] == 2
    assert summary["auto_store_precision"] == pytest.approx(1 / 3)


def test_routing_rates():
    summary = summarize(RECORDS)
    assert summary["straight_through_rate"] == pytest.approx(2 / 3)  # a, b of a, b, c
    assert summary["review_catch_rate"] == pytest.approx(2 / 3)  # d, f of d, e, f


def test_field_accuracy_and_absent_fields():
    summary = summarize(RECORDS)
    assert summary["field_accuracy"] == pytest.approx(7 / 10)
    assert summary["field_accuracy_by_format"]["pdf_scan"] == pytest.approx(1 / 2)
    assert summary["absent_fields_left_empty"] == pytest.approx(1 / 2)


def test_costs_use_the_longest_matching_price():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert cost_usd("gpt-4o-mini-2024-07-18", usage) == pytest.approx(0.75)
    assert cost_usd("gpt-6-luna", usage) == pytest.approx(0.60)
    assert cost_usd("some-unlisted-model", usage) is None


def test_report_names_each_silent_error_and_why():
    result = {"run": "test-model", "summary": summarize(RECORDS), "documents": RECORDS}
    report = markdown_report([result])
    assert "  - b, Scanned PDF: total_amount wrong" in report
    assert "  - e, Plain text: due_date filled absent" in report


# The runner, with the graph stubbed out.

LABEL = {
    "id": "inv-01", "file": "inv-01.txt", "format": "text", "doc_type": "invoice",
    "absent_field": None, "expected_route": "auto",
    "fields": {"vendor_name": "Acme Corp", "total_amount": 1250.0},
}


class RecordingGraph:
    """Stores what it was given and stops at human review."""

    def __init__(self):
        self.state: dict = {}

    def invoke(self, state, config):
        self.state = state

    def get_state(self, config):
        values = {"classification": {"doc_type": "invoice"},
                  "extraction": {"fields": {"vendor_name": "Acme Corp", "total_amount": 1250.0}}}
        return SimpleNamespace(values=values, next=("human_review",))


class CrashingGraph:
    """Makes one billed model call, then fails."""

    def invoke(self, state, config):
        reply = AIMessage(content="invoice", response_metadata={"model_name": "gpt-6-luna"},
                          usage_metadata={"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000})
        FakeMessagesListChatModel(responses=[reply]).invoke("classify this")
        raise RuntimeError("model timed out")


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    (tmp_path / "inv-01.txt").write_text("INVOICE\nVendor: Acme Corp\nTotal: $1,250.00\n")
    monkeypatch.setattr(run_eval, "DATA_DIR", tmp_path)


def test_the_file_name_does_not_give_the_classifier_the_answer(dataset, monkeypatch):
    graph = RecordingGraph()
    monkeypatch.setattr(run_eval, "build_graph", lambda checkpointer: graph)
    record = run_document(LABEL, "gpt-6-luna")
    assert graph.state["document_filename"] == "document.txt"
    assert record["route"] == "review"
    assert record["outcomes"] == {"vendor_name": "correct", "total_amount": "correct"}


def test_a_crashed_document_counts_against_the_model(dataset, monkeypatch):
    monkeypatch.setattr(run_eval, "build_graph", lambda checkpointer: CrashingGraph())
    record = run_document(LABEL, "gpt-6-luna")
    assert record["route"] == "error"
    assert record["error"] == "RuntimeError: model timed out"
    # Every field counts as not extracted, so the failure lowers accuracy...
    assert record["outcomes"] == {"vendor_name": "not_extracted", "total_amount": "not_extracted"}
    assert summarize([record])["field_accuracy"] == 0
    # ...and the tokens spent before the crash are still in the cost.
    assert record["input_tokens"] == 1_000_000
    assert record["cost_usd"] == pytest.approx(cost_usd("gpt-6-luna", {"input_tokens": 1_000_000}))
