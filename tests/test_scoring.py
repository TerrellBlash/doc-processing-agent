"""
TESTS — eval scoring rules
============================
Pure functions, no network.

Why these matter: published accuracy numbers are only as honest as the
comparison behind them. A scorer that's too strict reports formatting as
errors; one that's too loose hides real ones. The cases below pin the
line: formatting differences pass, value differences fail, and a value
for a field the document doesn't have is an error, even a placeholder,
because the confidence gate counts any value as present.

Run with: pytest tests/test_scoring.py -v
"""

import pytest

from evals.scoring import score_document, score_field


@pytest.mark.parametrize("printed", ["March 5, 2026", "2026-03-05", "05 Mar 2026", "Mar 5, 2026", "3/5/2026"])
def test_dates_match_in_any_printed_format(printed):
    assert score_field("date", "2026-03-05", printed) == "correct"


def test_a_different_date_is_wrong():
    assert score_field("date", "2026-03-05", "2026-05-03") == "wrong"


def test_amounts_ignore_formatting_but_not_value():
    assert score_field("amount", 19593.25, "$19,593.25") == "correct"
    assert score_field("amount", 19593.25, 19593.250000001) == "correct"
    assert score_field("amount", 19593.25, 19593.52) == "wrong"


def test_company_suffixes_and_case_are_ignored():
    assert score_field("company", "Brightline Logistics LLC", "brightline logistics") == "correct"
    assert score_field("company", "Brightline Logistics LLC", "Brightline Freight LLC") == "wrong"


def test_missing_value_for_a_field_the_document_has_is_missed():
    assert score_field("amount", 120.0, None) == "missed"
    assert score_field("names:company", ["Acme"], []) == "missed"


@pytest.mark.parametrize("actual", [None, "", "   ", []])
def test_empty_answer_for_an_absent_field_is_correct(actual):
    assert score_field("date", None, actual) == "correct"


@pytest.mark.parametrize("actual", ["Not specified", "N/A", "2026-04-04", "USD"])
def test_any_value_for_an_absent_field_is_an_error(actual):
    """The gate counts any non-empty value as present, so a placeholder
    passes it just like an invented value would."""
    assert score_field("currency", None, actual) == "filled_absent"


def test_phone_compares_digits():
    assert score_field("phone", "(404) 555-0142", "404.555.0142") == "correct"
    assert score_field("phone", "(404) 555-0142", "+1 404 555 0142") == "correct"
    assert score_field("phone", "(404) 555-0142", "(404) 555-0124") == "wrong"


def test_currency_symbol_maps_to_code():
    assert score_field("currency", "USD", "$") == "correct"
    assert score_field("currency", "EUR", "eur") == "correct"
    assert score_field("currency", "EUR", "USD") == "wrong"


def test_skills_need_most_of_the_list():
    expected = ["Python", "SQL", "Docker", "Kubernetes", "Terraform"]
    assert score_field("skills", expected, ["python", "sql", "docker", "kubernetes", "terraform", "git"]) == "correct"
    assert score_field("skills", expected, ["Python", "SQL"]) == "wrong"


def test_line_items_need_the_same_count_and_totals():
    expected = [{"total": 100.0}, {"total": 250.5}]
    assert score_field("line_items", expected, [{"total": 250.5}, {"total": 100.0}]) == "correct"
    assert score_field("line_items", expected, [{"total": 350.5}]) == "wrong"


def test_names_compare_as_sets_from_the_given_key():
    expected = ["Harbor & Finch Supply Co.", "Redwood Data Systems"]
    extracted = [{"name": "Redwood Data Systems, Inc."}, {"name": "Harbor & Finch Supply"}]
    assert score_field("names:name", expected, extracted) == "correct"


def test_an_entry_without_a_name_neither_helps_nor_hurts():
    """A job listed with no employer is an empty entry, not a company
    called "None"."""
    expected = ["Lumen Freight", "Polaris Payments"]
    extracted = [{"company": "Lumen Freight"}, {"company": None}, {"company": "Polaris Payments"}]
    assert score_field("names:company", expected, extracted) == "correct"
    assert score_field("names:company", expected, [{"company": "Lumen Freight"}, {"company": None}]) == "wrong"


def test_contract_type_accepts_common_names():
    assert score_field("contract_type", "Master Services Agreement", "MSA") == "correct"
    assert score_field("contract_type", "Consulting Agreement", "Independent Consulting Agreement") == "correct"
    assert score_field("contract_type", "Consulting Agreement", "Lease") == "wrong"


def test_jurisdiction_and_location_match_on_the_place_name():
    assert score_field("place", "Georgia", "State of Georgia") == "correct"
    assert score_field("place", "Austin", "Austin, Texas") == "correct"
    assert score_field("place", "Georgia", "Delaware") == "wrong"


def test_presence_fields_only_need_some_text():
    assert score_field("presence", "Either party may terminate...", "30 days notice") == "correct"


def test_document_with_no_extraction_scores_every_field_as_not_extracted():
    expected = {"vendor_name": "Acme Corp", "due_date": None}
    outcomes = score_document("invoice", expected, None)
    assert outcomes == {"vendor_name": "not_extracted", "due_date": "not_extracted"}
