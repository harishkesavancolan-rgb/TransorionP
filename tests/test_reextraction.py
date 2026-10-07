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


def test_country_guardrail_does_not_rewrite_import_origin(monkeypatch):
    # extract_and_validate must pass its shipment_type through to
    # apply_header_guardrails: on an import, destination is always India,
    # so the export-only origin correction must not turn a real foreign
    # origin into "IN" just because the supplier address mentions India.
    import_schema = load_template_schema(TEMPLATE_PATH["import"])
    header = {
        **_valid_header(),
        "supplier_address": "Liaison Office: 12 MG Road, Bengaluru, India",
        "country_of_origin": "CHINA", "country_of_destination": "INDIA",
    }
    monkeypatch.setattr(reextraction, "extract_invoice_header", _Sequenced([header]))
    monkeypatch.setattr(reextraction, "extract_line_items", _Sequenced([[_valid_item()]]))

    result = _run(reextraction.extract_and_validate(
        "INVOICE TEXT", import_schema, "import", max_retries=0,
    ))

    assert result["header"]["country_of_origin"] == "CHINA"


# ── guardrail notes + corrections survive into extract_and_validate ─────

def test_ocr_variant_invoice_number_does_not_double_the_total_end_to_end(monkeypatch):
    # Real case: "SI3626204603 / S13626204603", totals "30875.00 / 30875.00".
    # The validator used to sum those to 61,750 and fail HEADER_LINE_TOTAL.
    header = {
        "invoice_number": "SI3626204603 / S13626204603", "invoice_date": "2026-08-26 / 2026-08-26",
        "invoice_total": "100.00 / 100.00", "currency": "USD",
    }
    monkeypatch.setattr(reextraction, "extract_invoice_header", _Sequenced([header]))
    monkeypatch.setattr(reextraction, "extract_line_items", _Sequenced([[_valid_item()]]))

    result = _run(reextraction.extract_and_validate("INVOICE TEXT", EXPORT_SCHEMA, "export", max_retries=0))

    assert result["extraction_status"] == "validated"
    assert result["header"]["invoice_number"] == "SI3626204603"
    assert any("collapsed" in w for w in result["warnings"])


def test_misread_line_total_is_repaired_from_the_source_text_end_to_end(monkeypatch):
    item = dict(_valid_item(), quantity=10, unit_price=10.0, line_total=90.0)  # printed amount is 100.00
    monkeypatch.setattr(reextraction, "extract_invoice_header", _Sequenced([_valid_header()]))
    monkeypatch.setattr(reextraction, "extract_line_items", _Sequenced([[item]]))

    result = _run(reextraction.extract_and_validate(
        "10 PCS   10.00   100.00\nTotal 100.00", EXPORT_SCHEMA, "export", max_retries=0,
    ))

    assert result["extraction_status"] == "validated"
    assert result["sheets"]["ITEM"][0]["ItmTaxableVal"] == 100.0
    assert any("replaced with 100.00" in w for w in result["warnings"])


# ── missing rows: one page-by-page retry, header never touched ──────────

def _two_item_total_header():
    return dict(_valid_header(), invoice_total=200.0)


def test_incomplete_item_list_triggers_one_page_by_page_retry_that_fixes_it(monkeypatch):
    # Real case: the printed total is bigger than the extracted lines because
    # the model stopped after one page of rows.
    header_calls = _Sequenced([_two_item_total_header()])
    items_calls = _Sequenced([[_valid_item()], [_valid_item(), dict(_valid_item(), item_ser_no=2)]])
    monkeypatch.setattr(reextraction, "extract_invoice_header", header_calls)
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate("TEXT", EXPORT_SCHEMA, "export", max_retries=2))

    assert result["extraction_status"] == "validated" and result["attempts"] == 2
    assert len(result["sheets"]["ITEM"]) == 2
    assert items_calls.calls[0].get("fine_chunks") is None
    assert items_calls.calls[1].get("fine_chunks") is True  # the page-by-page retry
    assert len(header_calls.calls) == 1  # the HEADER is never re-extracted


def test_completeness_retry_is_spent_only_once_and_never_replaces_a_better_list(monkeypatch):
    # The retry comes back WORSE (no items): keep the previous list, and do
    # not ask again (a third items call would raise in _Sequenced).
    monkeypatch.setattr(reextraction, "extract_invoice_header", _Sequenced([_two_item_total_header()]))
    items_calls = _Sequenced([[_valid_item()], []])
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate("TEXT", EXPORT_SCHEMA, "export", max_retries=2))

    assert result["extraction_status"] == "validation_failed"
    assert len(result["sheets"]["ITEM"]) == 1  # the earlier, better list survived
    assert len(items_calls.calls) == 2
    assert any(e["rule"] == "LINE_ITEMS_INCOMPLETE" for e in result["validation"]["errors"])


def test_any_items_retry_never_replaces_a_list_with_one_further_from_the_printed_total(monkeypatch):
    # Real case (Flowserve 20262201923): attempt 2 had every row and the
    # exact printed total but one misread quantity (QTY_X_UNIT_PRICE); the
    # feedback retry for that came back with rows missing, and the worse
    # list became the final answer. Printed total 200: the first list sums
    # to 200 with one wrong quantity, the retry sums to only 100.
    wrong_qty = dict(_valid_item(), item_ser_no=2, quantity=3)  # 3 x 10.00 != 100.00 -> retryable error
    first = [_valid_item(), wrong_qty]
    worse = [_valid_item()]
    items_calls = _Sequenced([first, worse])
    monkeypatch.setattr(reextraction, "extract_invoice_header", _Sequenced([_two_item_total_header()]))
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate("TEXT", EXPORT_SCHEMA, "export", max_retries=1))

    assert len(items_calls.calls) == 2  # the retry did run
    assert len(result["sheets"]["ITEM"]) == 2  # ...but its worse list was not kept


def test_lines_over_the_printed_total_are_never_retried(monkeypatch):
    header_calls = _Sequenced([dict(_valid_header(), invoice_total=100.0)])
    other = dict(_valid_item(), item_ser_no=2, product_description="A DIFFERENT PRODUCT", quantity=5,
                 unit_price=20.0, line_total=100.0)  # a distinct line, so it isn't a bundle duplicate
    items_calls = _Sequenced([[_valid_item(), other]])  # lines sum to 200 against a printed total of 100
    monkeypatch.setattr(reextraction, "extract_invoice_header", header_calls)
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate("TEXT", EXPORT_SCHEMA, "export", max_retries=2))

    assert result["extraction_status"] == "validation_failed" and result["attempts"] == 1
    assert len(items_calls.calls) == 1 and len(header_calls.calls) == 1


# ── HSN end to end ──────────────────────────────────────────────────────

_ROW_TEXT = "1 │ 160 │ 8481 80 90 │ 402/Y3499 A REAL DESCRIPTION │ 10 │ NOS │ 10.00 │ 100.00\n"


def test_item_number_in_the_hsn_field_is_repaired_from_the_source_row(monkeypatch):
    item = dict(_valid_item(), hsn_code="160", part_number="402/Y3499")
    monkeypatch.setattr(reextraction, "extract_invoice_header", _Sequenced([_valid_header()]))
    monkeypatch.setattr(reextraction, "extract_line_items", _Sequenced([[item]]))

    result = _run(reextraction.extract_and_validate(_ROW_TEXT, EXPORT_SCHEMA, "export", max_retries=0))

    assert result["extraction_status"] == "validated"
    assert result["sheets"]["ITEM"][0]["Item_RITC"] == "8481 80 90"
    assert any("hsn_code '160'" in w for w in result["warnings"])


def test_unrepairable_item_number_in_hsn_goes_back_to_the_model_with_a_clear_message(monkeypatch):
    bad = dict(_valid_item(), hsn_code="160")
    good = dict(_valid_item(), hsn_code="8481 80 90")
    items_calls = _Sequenced([[bad], [good]])
    monkeypatch.setattr(reextraction, "extract_invoice_header", _Sequenced([_valid_header()]))
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate("NO ROW TEXT HERE", EXPORT_SCHEMA, "export", max_retries=2))

    assert result["extraction_status"] == "validated" and result["attempts"] == 2
    assert result["sheets"]["ITEM"][0]["Item_RITC"] == "8481 80 90"
    assert "HSN Code" in items_calls.calls[1]["feedback"]


# ── the completeness retry judges a list AFTER repairing it ─────────────

def test_a_complete_list_with_a_repairable_wrong_column_amount_beats_a_short_list(monkeypatch):
    # Real case (227647): the page-by-page retry returned ALL rows, but one
    # row's raw amounts came from the wrong column, putting the raw sum
    # FURTHER from the printed total than the short first list -- yet the
    # repair from the printed row fixes it exactly.
    row1 = "1 │ 10 │ 8481 80 90 │ AAA-1111 WIDGET ASSEMBLY │ 2 │ NOS │ 25.00 │ 50.00 │ 50.00\n"
    row2 = "2 │ 20 │ 8481 80 90 │ BBB-2222 GADGET ASSEMBLY │ 4 │ NOS │ 12.50 │ 50.00 │ 50.00\n"
    text = row1 + row2
    first = [dict(_valid_item(), product_description="WIDGET ASSEMBLY", part_number="AAA-1111",
                  quantity=2, unit_price=25.0, line_total=50.0)]
    retry = [
        dict(first[0]),
        # wrong columns: the amount taken as the unit price, a derived total
        dict(_valid_item(), item_ser_no=2, product_description="GADGET ASSEMBLY", part_number="BBB-2222",
             quantity=4, unit_price=50.0, line_total=200.0),
    ]
    header = dict(_valid_header(), invoice_total=100.0)
    items_calls = _Sequenced([first, retry])
    monkeypatch.setattr(reextraction, "extract_invoice_header", _Sequenced([header]))
    monkeypatch.setattr(reextraction, "extract_line_items", items_calls)

    result = _run(reextraction.extract_and_validate(text, EXPORT_SCHEMA, "export", max_retries=2))

    assert result["extraction_status"] == "validated"
    rows = result["sheets"]["ITEM"]
    assert [(r["Item_Unit_Price"], r["ItmTaxableVal"]) for r in rows] == [(25.0, 50.0), (12.5, 50.0)]
    assert any("corrected to unit_price 12.5" in w for w in result["warnings"])  # the repair note survived


def test_a_stray_row_with_no_numbers_does_not_make_the_retry_look_worse(monkeypatch):
    # Real case (CIV, 214838): the retry returned the right rows plus one
    # fragment with no quantity/price/amount.
    first = [_valid_item()]
    stray = {"item_ser_no": None, "product_description": "STRAY FRAGMENT", "quantity": None, "unit_price": None, "line_total": None}
    retry = [_valid_item(), dict(_valid_item(), item_ser_no=2), stray]
    header = dict(_valid_header(), invoice_total=200.0)
    monkeypatch.setattr(reextraction, "extract_invoice_header", _Sequenced([header]))
    monkeypatch.setattr(reextraction, "extract_line_items", _Sequenced([first, retry]))

    result = _run(reextraction.extract_and_validate("TEXT", EXPORT_SCHEMA, "export", max_retries=2))

    assert result["extraction_status"] == "validated"
    assert len(result["sheets"]["ITEM"]) == 2
    assert any("discarded a row with no quantity, unit price or amount" in w for w in result["warnings"])


def test_notes_of_the_kept_list_survive_when_a_worse_retry_is_rejected(monkeypatch):
    row = "1 │ 10 │ 8481 80 90 │ AAA-1111 WIDGET ASSEMBLY │ 2 │ NOS │ 25.00 │ 50.00 │ 50.00\n"
    first = [dict(_valid_item(), product_description="WIDGET ASSEMBLY", part_number="AAA-1111",
                  quantity=2, unit_price=50.0, line_total=100.0, hsn_code="10")]  # item number in the HSN field
    header = dict(_valid_header(), invoice_total=300.0)  # first list stays short -> a retry happens
    monkeypatch.setattr(reextraction, "extract_invoice_header", _Sequenced([header]))
    monkeypatch.setattr(reextraction, "extract_line_items", _Sequenced([first, []]))

    result = _run(reextraction.extract_and_validate(row, EXPORT_SCHEMA, "export", max_retries=2))

    assert result["extraction_status"] == "validation_failed"
    assert any("hsn_code '10'" in w for w in result["warnings"])  # the kept list's repair note is still reported
