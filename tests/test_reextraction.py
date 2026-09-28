"""
Tests for reextraction.py's retry orchestration. The LLM calls
(llm_extract.extract_invoice_header / extract_line_items, imported by
name into reextraction.py) are mocked so these tests run instantly and
free -- they check the RETRY POLICY (attempt counting, targeted
re-extraction, stopping conditions), not model output quality, which is
invoice_validator.py's job and already covered separately.

Uses asyncio.run() directly rather than pytest-asyncio (not an existing
project dependency, and one plain helper avoids adding one just for this).
"""
from __future__ import annotations

import asyncio

import reextraction
from schema import load_template_schema
from templates_config import TEMPLATE_PATH

EXPORT_SCHEMA = load_template_schema(TEMPLATE_PATH["export"])

_USAGE = {"input_tokens": 10, "cached_tokens": 0, "output_tokens": 10}


def _valid_header():
    return {"invoice_total": 100.0, "invoice_number": "INV-1", "invoice_date": "2026-08-08"}


def _valid_item():
    return {
        "item_ser_no": 1, "product_description": "A REAL DESCRIPTION", "quantity": 10,
        "unit_of_measurement": "PCS", "unit_price": 10.0, "line_total": 100.0,
    }


def _invalid_item():
    # Missing description (REQUIRED_FIELD_MISSING) -- deliberately keeps
    # quantity/unit_price/line_total consistent with _valid_item() and
    # with the header's invoice_total (100 = 10 x 10 either way), so this
    # only fails an item-scoped rule and leaves HEADER_LINE_TOTAL PASSing
    # in both the invalid and the fixed state -- needed to cleanly test
    # that a header-unrelated failure doesn't trigger a header re-extract.
    return {
        "item_ser_no": 1, "product_description": "", "quantity": 10,
        "unit_of_measurement": "PCS", "unit_price": 10.0, "line_total": 100.0,
    }


class _Sequenced:
    """Returns each entry in `responses` in order, one per call; raises if
    called more times than there are responses (keeps a test's assertion
    about call COUNT honest -- an unexpected extra call fails loudly
    instead of silently reusing the last response)."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("called more times than expected")
        resp = self.responses.pop(0)
        return resp, dict(_USAGE)


def _run(coro):
    return asyncio.run(coro)


# ── 14 & 16: retry after failure, then succeed ──────────────────────────

def test_retry_after_validation_failure_then_succeeds(monkeypatch):
    header_calls = _Sequenced([_valid_header()])  # header is correct from the start
    items_calls = _Sequenced([[_invalid_item()], [_valid_item()]])  # items pass fails once, then succeeds

    monkeypatch.setattr(reextraction, "extract_invoice_header", header_calls)
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate(
        "INVOICE TEXT", EXPORT_SCHEMA, "export", max_retries=2,
    ))

    assert result["extraction_status"] == "validated"
    assert result["attempts"] == 2
    assert len(result["validation_history"]) == 2
    assert result["validation_history"][0]["valid"] is False
    assert result["validation_history"][1]["valid"] is True

    # Targeted re-extraction: header pass only ran ONCE (its attempt-1
    # output already validated), items pass ran twice.
    assert len(header_calls.calls) == 1
    assert len(items_calls.calls) == 2
    # the retry call carried feedback and the previous attempt's items
    assert items_calls.calls[1]["feedback"] is not None
    assert items_calls.calls[1]["previous_result"] == [_invalid_item()]


# ── 15 & 17: maximum retry limit, still invalid -> stop and report failure ──

def test_maximum_retry_limit_stops_and_reports_failure(monkeypatch):
    header_calls = _Sequenced([_valid_header()])
    # every items attempt is invalid -- 1 initial + 2 retries = 3 total
    items_calls = _Sequenced([[_invalid_item()], [_invalid_item()], [_invalid_item()]])

    monkeypatch.setattr(reextraction, "extract_invoice_header", header_calls)
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate(
        "INVOICE TEXT", EXPORT_SCHEMA, "export", max_retries=2,
    ))

    assert result["extraction_status"] == "validation_failed"
    assert result["attempts"] == 3
    assert len(result["validation_history"]) == 3
    assert all(h["valid"] is False for h in result["validation_history"])
    assert result["validation"]["valid"] is False
    assert len(result["validation"]["errors"]) > 0

    # exactly 3 attempts were made -- no infinite loop, no 4th call
    assert len(items_calls.calls) == 3
    assert len(header_calls.calls) == 1  # header never needed a retry


def test_zero_max_retries_makes_exactly_one_attempt(monkeypatch):
    header_calls = _Sequenced([_valid_header()])
    items_calls = _Sequenced([[_invalid_item()]])

    monkeypatch.setattr(reextraction, "extract_invoice_header", header_calls)
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate(
        "INVOICE TEXT", EXPORT_SCHEMA, "export", max_retries=0,
    ))

    assert result["attempts"] == 1
    assert result["extraction_status"] == "validation_failed"
    assert len(items_calls.calls) == 1


def test_valid_on_first_attempt_needs_no_retry(monkeypatch):
    header_calls = _Sequenced([_valid_header()])
    items_calls = _Sequenced([[_valid_item()]])

    monkeypatch.setattr(reextraction, "extract_invoice_header", header_calls)
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate(
        "INVOICE TEXT", EXPORT_SCHEMA, "export", max_retries=2,
    ))

    assert result["attempts"] == 1
    assert result["extraction_status"] == "validated"
    assert len(header_calls.calls) == 1
    assert len(items_calls.calls) == 1


def test_header_only_failure_does_not_reextract_items(monkeypatch):
    # header missing invoice_number (REQUIRED_FIELD_MISSING, retryable); items correct
    bad_header = {"invoice_total": 100.0, "invoice_number": "", "invoice_date": "2026-08-08"}
    good_header = _valid_header()
    header_calls = _Sequenced([bad_header, good_header])
    items_calls = _Sequenced([[_valid_item()]])  # only ever called once

    monkeypatch.setattr(reextraction, "extract_invoice_header", header_calls)
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate(
        "INVOICE TEXT", EXPORT_SCHEMA, "export", max_retries=2,
    ))

    assert result["extraction_status"] == "validated"
    assert result["attempts"] == 2
    assert len(header_calls.calls) == 2
    assert len(items_calls.calls) == 1  # items pass reused, never re-called


# ── apply_header_guardrails must actually run in production (regression) ─

def test_header_guardrails_are_applied_through_extract_and_validate(monkeypatch):
    # Confirmed real gap: apply_header_guardrails() used to be inline code
    # inside a since-removed llm_extract.extract_fields() convenience
    # wrapper, which reextraction.py (the actual production entry point
    # every real run goes through) never called -- so none of those
    # guardrails ever fired in production. This locks in the fix: the
    # model hallucinates a net_realisable_amount even
    # though the source text never says "net realisable"/"net charge" --
    # the guardrail must null it out, and that correction must survive all
    # the way through to extract_and_validate()'s returned header.
    bad_header = {
        "invoice_total": 100.0, "invoice_number": "INV-1", "invoice_date": "2026-08-08",
        "net_realisable_amount": 999.0,
    }
    header_calls = _Sequenced([bad_header])
    items_calls = _Sequenced([[_valid_item()]])

    monkeypatch.setattr(reextraction, "extract_invoice_header", header_calls)
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    invoice_text = "INVOICE TEXT with no mention of that concept at all"
    result = _run(reextraction.extract_and_validate(
        invoice_text, EXPORT_SCHEMA, "export", max_retries=2,
    ))

    assert result["header"]["net_realisable_amount"] == ""
