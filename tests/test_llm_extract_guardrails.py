"""
Tests for llm_extract.apply_header_guardrails() (the deterministic
post-processing guardrails run after both LLM passes) and
extract_line_items_chunked() (the page-chunked line-items pass).
apply_header_guardrails() is synchronous and mutates its header dict in
place, so it's exercised directly, no mocking needed; the chunked
extraction's own LLM call (extract_line_items) is mocked so those tests
run instantly and free -- they check the chunking/merge logic, not model
output quality.

Uses asyncio.run() directly rather than pytest-asyncio, same as
test_reextraction.py.
"""
from __future__ import annotations

import asyncio

import llm_extract

_USAGE = {"input_tokens": 10, "cached_tokens": 0, "output_tokens": 10}


# ── total_packages: narrative packing-list fallback ─────────────────────

def test_total_packages_recovered_from_narrative_carton_description():
    # Model copied the piece quantity (3840) into total_packages instead
    # of the real package count (192 carton boxes); 4 is a pallet count
    # that must NOT be picked up either.
    text = "Four Nos Of Pallet Containing 3840 Nos (20 Nos Each in 192 carton boxes)\n"
    header = {"total_packages": 3840}
    items = [{"item_ser_no": 1, "quantity": 3840}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["total_packages"] == 192


def test_total_packages_recovered_when_missing_entirely():
    text = "Packed in 45 cartons, net weight 900 KG\n"
    header = {"total_packages": None}
    items = [{"item_ser_no": 1, "quantity": 900}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["total_packages"] == 45


def test_total_packages_left_alone_when_already_different_and_plausible():
    # Not the quantity and not missing -- an already-distinct value is
    # trusted rather than second-guessed against text.
    text = "Packed in 45 cartons\n"
    header = {"total_packages": 12}
    items = [{"item_ser_no": 1, "quantity": 900}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["total_packages"] == 12


def test_total_packages_not_guessed_from_ambiguous_multiple_candidates():
    # Two different, conflicting package-unit mentions -- no way to pick
    # the right one from shape alone, so it's left as-is rather than
    # guessed.
    text = "192 cartons on pallet A, 50 cases on pallet B\n"
    header = {"total_packages": 3840}
    items = [{"item_ser_no": 1, "quantity": 3840}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["total_packages"] == 3840


def test_total_packages_ignores_pallet_count():
    # No carton/case/box/package wording at all, only a pallet count --
    # must NOT be substituted in as total_packages.
    text = "Shipped as 4 pallets\n"
    header = {"total_packages": 3840}
    items = [{"item_ser_no": 1, "quantity": 3840}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["total_packages"] == 3840


# ── country_of_origin: export-only origin/destination swap fix ──────────

def _swapped_origin_header(supplier_address):
    return {
        "supplier_address": supplier_address,
        "country_of_origin": "GERMANY", "country_of_destination": "GERMANY",
    }


def test_export_origin_equal_to_destination_is_corrected_to_india():
    header = _swapped_origin_header("Plot 4, MIDC, Pune 411019, Maharashtra, India")
    llm_extract.apply_header_guardrails(header, [], "", shipment_type="export")
    assert header["country_of_origin"] == "IN"


def test_import_origin_is_never_rewritten_even_if_supplier_mentions_india():
    # Import destination is always India, so the "dest == IN" swap case
    # would fire on every import whose supplier address names India.
    header = {
        "supplier_address": "Shenzhen, China (India liaison office: Bengaluru, India)",
        "country_of_origin": "CHINA", "country_of_destination": "INDIA",
    }
    llm_extract.apply_header_guardrails(header, [], "", shipment_type="import")
    assert header["country_of_origin"] == "CHINA"


def test_supplier_in_indiana_is_not_treated_as_indian():
    header = {
        "supplier_address": "500 Main St, Indianapolis, Indiana 46204, USA",
        "country_of_origin": "USA", "country_of_destination": "INDIA",
    }
    llm_extract.apply_header_guardrails(header, [], "", shipment_type="export")
    assert header["country_of_origin"] == "USA"


def test_country_guardrail_skipped_when_shipment_type_not_given():
    header = _swapped_origin_header("Plot 4, MIDC, Pune 411019, Maharashtra, India")
    llm_extract.apply_header_guardrails(header, [], "")
    assert header["country_of_origin"] == "GERMANY"


def test_apply_header_guardrails_is_a_noop_on_a_non_dict_header():
    # Defensive: a malformed LLM response (header came back as something
    # other than a dict) must not raise.
    llm_extract.apply_header_guardrails(None, [], "some text\n")
    llm_extract.apply_header_guardrails("not a dict", [], "some text\n")


# ── _split_text_by_page ──────────────────────────────────────────────────

def _page_marker(idx, total, name="inv.pdf"):
    return f"\n{'=' * 35} PAGE {idx} OF {total} ({name}) {'=' * 35}\n"


def test_split_text_by_page_splits_on_markers():
    text = (
        _page_marker(1, 3) + "page one content\n"
        + _page_marker(2, 3) + "page two content\n"
        + _page_marker(3, 3) + "page three content\n"
    )
    pages = llm_extract._split_text_by_page(text)
    assert len(pages) == 3
    assert "PAGE 1 OF 3" in pages[0] and "page one content" in pages[0]
    assert "PAGE 2 OF 3" in pages[1] and "page two content" in pages[1]
    assert "PAGE 3 OF 3" in pages[2] and "page three content" in pages[2]


def test_split_text_by_page_single_page_returns_unchanged():
    text = _page_marker(1, 1) + "only page\n"
    pages = llm_extract._split_text_by_page(text)
    assert pages == [text]


def test_split_text_by_page_no_markers_returns_unchanged():
    text = "plain text with no page markers at all\n"
    pages = llm_extract._split_text_by_page(text)
    assert pages == [text]


# ── extract_line_items_chunked ───────────────────────────────────────────

def test_chunked_extraction_below_threshold_makes_a_single_call(monkeypatch):
    calls = []

    async def fake_extract_line_items(text, **kwargs):
        calls.append(text)
        return [{"item_ser_no": 1}], dict(_USAGE)

    monkeypatch.setattr(llm_extract, "extract_line_items", fake_extract_line_items)

    text = (
        _page_marker(1, 2) + "page one\n"
        + _page_marker(2, 2) + "page two\n"
    )
    items, usage = asyncio.run(llm_extract.extract_line_items_chunked(text, model="gpt-5-nano"))
    assert len(calls) == 1  # 2 pages <= threshold -- single call, whole text
    assert calls[0] == text
    assert items == [{"item_ser_no": 1}]
    assert usage == _USAGE


def test_chunked_extraction_above_threshold_splits_and_merges_per_page(monkeypatch):
    import re

    calls = []

    async def fake_extract_line_items(text, **kwargs):
        calls.append(text)
        # A chunk may cover _LINE_ITEMS_PAGES_PER_CHUNK pages at once --
        # return one item per page marker actually present in this call's
        # text, not one item per call, so merging across chunks is what
        # gets tested here (chunk boundaries, not per-page granularity).
        page_nums = [int(n) for n in re.findall(r"PAGE (\d+) OF", text)]
        return (
            [{"item_ser_no": n, "product_description": f"item from page {n}"} for n in page_nums],
            dict(_USAGE),
        )

    monkeypatch.setattr(llm_extract, "extract_line_items", fake_extract_line_items)

    n_pages = 5  # > _LINE_ITEMS_CHUNK_PAGE_THRESHOLD (3)
    text = "".join(_page_marker(i, n_pages) + f"content {i}\n" for i in range(1, n_pages + 1))

    items, usage = asyncio.run(llm_extract.extract_line_items_chunked(text, model="gpt-5-nano"))

    expected_n_chunks = -(-n_pages // llm_extract._LINE_ITEMS_PAGES_PER_CHUNK)  # ceil division
    assert len(calls) == expected_n_chunks  # grouped into chunks, not one call per page
    assert [it["item_ser_no"] for it in items] == [1, 2, 3, 4, 5]  # still merged in page order
    # Usage summed across all chunk calls, not just the last one.
    assert usage["input_tokens"] == _USAGE["input_tokens"] * expected_n_chunks
    assert usage["output_tokens"] == _USAGE["output_tokens"] * expected_n_chunks


def test_chunked_extraction_passes_feedback_to_every_chunk(monkeypatch):
    seen_feedback = []

    async def fake_extract_line_items(text, feedback=None, **kwargs):
        seen_feedback.append(feedback)
        return [], dict(_USAGE)

    monkeypatch.setattr(llm_extract, "extract_line_items", fake_extract_line_items)

    n_pages = 4
    text = "".join(_page_marker(i, n_pages) + f"content {i}\n" for i in range(1, n_pages + 1))
    asyncio.run(llm_extract.extract_line_items_chunked(text, model="gpt-5-nano", feedback="fix the totals"))

    expected_n_chunks = -(-n_pages // llm_extract._LINE_ITEMS_PAGES_PER_CHUNK)  # ceil division
    assert seen_feedback == ["fix the totals"] * expected_n_chunks


# ── _group_pages_into_chunks ──────────────────────────────────────────────

def test_group_pages_into_chunks_groups_by_size():
    pages = ["p1", "p2", "p3", "p4", "p5"]
    assert llm_extract._group_pages_into_chunks(pages, 2) == ["p1\n\np2", "p3\n\np4", "p5"]


def test_group_pages_into_chunks_size_one_is_noop():
    pages = ["p1", "p2", "p3"]
    assert llm_extract._group_pages_into_chunks(pages, 1) is pages


# ── invoice_total vs "F.O.B Value" guardrail ────────────────────────────
# Must only discard invoice_total when there's EVIDENCE the FOB line is a
# different currency -- on an FOB-terms invoice, FOB value == total is
# exactly what a correct invoice looks like.

def test_fob_terms_invoice_keeps_its_correct_total():
    # Same currency, items sum to the total: nothing wrong, nothing to null.
    text = "Incoterms: FOB Chennai\nFOB Value: 5,000.00\nTotal Invoice Value USD 5,000.00\n"
    header = {"invoice_total": 5000.0, "currency": "USD"}
    items = [{"line_total": 3000.0}, {"line_total": 2000.0}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["invoice_total"] == 5000.0


def test_fob_equal_to_total_is_left_alone_without_any_evidence():
    # No line amounts and no INR label: can't tell, so don't delete.
    text = "FOB Value: 5,000.00\n"
    header = {"invoice_total": 5000.0, "currency": "USD"}
    llm_extract.apply_header_guardrails(header, [], text)
    assert header["invoice_total"] == 5000.0


def test_fob_line_labeled_inr_on_a_usd_invoice_discards_the_total():
    text = "Total F.O.B Value (INR): 4,25,000.00\n"
    header = {"invoice_total": 425000.0, "currency": "USD"}
    llm_extract.apply_header_guardrails(header, [], text)
    assert header["invoice_total"] is None


def test_fob_line_labeled_rs_prefers_net_realisable_when_present():
    text = "FOB Value Rs. 4,25,000.00\nNet Realisable Amount 5,000.00\n"
    header = {"invoice_total": 425000.0, "currency": "USD", "net_realisable_amount": 5000.0}
    llm_extract.apply_header_guardrails(header, [], text)
    assert header["invoice_total"] == 5000.0


def test_fob_unlabeled_but_items_contradict_total_discards_it():
    # The originally-confirmed case: items sum to the USD total; the
    # model's invoice_total is the larger INR FOB figure.
    text = "FOB Value 425,000.00\n"
    header = {"invoice_total": 425000.0, "currency": "USD"}
    items = [{"line_total": 3000.0}, {"line_total": 2000.0}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["invoice_total"] is None


def test_fob_on_an_inr_invoice_is_never_discarded():
    text = "FOB Value (INR): 4,25,000.00\n"
    header = {"invoice_total": 425000.0, "currency": "INR"}
    llm_extract.apply_header_guardrails(header, [], text)
    assert header["invoice_total"] == 425000.0


def test_partial_line_amounts_do_not_count_as_contradicting_the_total():
    # One item has no amount, so the sum (2000) is incomplete -- it can't
    # be used as evidence against a total that also covers the missing item.
    text = "FOB Value: 5,000.00\n"
    header = {"invoice_total": 5000.0, "currency": "USD"}
    items = [{"line_total": 2000.0}, {"line_total": None}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["invoice_total"] == 5000.0


def test_fob_guardrail_tolerates_a_combined_or_non_numeric_total():
    text = "FOB Value: 5,000.00\n"
    header = {"invoice_total": "4416.00 / 1472.00", "currency": "USD"}
    llm_extract.apply_header_guardrails(header, [], text)  # must not raise
    assert header["invoice_total"] == "4416.00 / 1472.00"


# ── same invoice number, spelled two ways by OCR, is ONE invoice ────────
# Confirmed real cases (two Hetero invoices): the number is printed on
# every page and OCR read a letter I as the digit 1 on some pages.

def test_ocr_variant_of_one_invoice_number_collapses_to_one_invoice():
    header = {
        "invoice_number": "S13726201956 / SI3726201956",
        "invoice_date": "2026-08-28 / 2026-08-28",
        "invoice_total": "3120.00 / 3120.00",
    }
    notes = []
    llm_extract.apply_header_guardrails(header, [], "", notes=notes)
    assert header["invoice_number"] == "SI3726201956"  # the spelling with the letter I
    assert header["invoice_date"] == "2026-08-28"
    assert header["invoice_total"] == "3120.00"  # not "3120.00 / 3120.00", which sums to 6240
    assert notes and "collapsed" in notes[0]


def test_second_hetero_case_collapses_and_keeps_the_letter_spelling():
    header = {
        "invoice_number": "SI3626204603 / S13626204603",
        "invoice_date": "2026-08-26 / 2026-08-26",
        "invoice_total": "30875.00 / 30875.00",
    }
    llm_extract.apply_header_guardrails(header, [], "")
    assert header["invoice_number"] == "SI3626204603"
    assert header["invoice_total"] == "30875.00"


def test_single_total_for_duplicate_numbers_is_left_as_is():
    # The model's attempt-1 answer: both spellings, but one (correct) total.
    header = {"invoice_number": "S13726201956 / SI3726201956", "invoice_total": "3120"}
    llm_extract.apply_header_guardrails(header, [], "")
    assert header["invoice_number"] == "SI3726201956"
    assert header["invoice_total"] == "3120"


def test_genuinely_different_sub_invoices_are_never_merged():
    header = {
        "invoice_number": "KA/2627/I/000431 / KA/2627/I/000433",
        "invoice_total": "4416.00 / 1472.00",
    }
    llm_extract.apply_header_guardrails(header, [], "")
    assert header["invoice_number"] == "KA/2627/I/000431 / KA/2627/I/000433"
    assert header["invoice_total"] == "4416.00 / 1472.00"


def test_duplicate_numbers_with_conflicting_totals_are_left_for_the_validator():
    header = {"invoice_number": "SI1001 / S11001", "invoice_total": "100.00 / 200.00"}
    llm_extract.apply_header_guardrails(header, [], "")
    assert header["invoice_number"] == "SI1001 / S11001"


def test_a_single_number_containing_slashes_is_untouched():
    header = {"invoice_number": "KA/2627/I/033552", "invoice_total": 100.0}
    llm_extract.apply_header_guardrails(header, [], "")
    assert header["invoice_number"] == "KA/2627/I/033552"


# ── line_total: repaired only when the page itself prints the right value ─

_PRINTED = "475.000 KG Net   65.00   30,875.00\nUSD 30,875.00\n"


def test_misread_line_total_is_replaced_by_the_amount_printed_on_the_page():
    # Real case: page prints 30,875.00; the model returned 30775.
    items = [{"quantity": 475, "unit_price": 65.0, "line_total": 30775}]
    notes = llm_extract.apply_line_item_guardrails(items, _PRINTED)
    assert items[0]["line_total"] == 30875.0
    assert notes and "30775" in notes[0] and "30875" in notes[0]


def test_line_total_that_the_document_itself_prints_is_never_replaced():
    # A genuine printed mismatch (the Piramal-style case): leave it for the
    # validator to flag rather than "fix" it.
    items = [{"quantity": 475, "unit_price": 65.0, "line_total": 30775}]
    text = "475.000 KG   65.00   30,775.00\nUSD 30,875.00\n"
    assert llm_extract.apply_line_item_guardrails(items, text) == []
    assert items[0]["line_total"] == 30775


def test_line_total_not_replaced_when_the_correct_amount_is_not_printed():
    # qty x rate isn't on the page anywhere -> nothing to support a change.
    items = [{"quantity": 475, "unit_price": 65.0, "line_total": 30775}]
    assert llm_extract.apply_line_item_guardrails(items, "475 KG 65.00\n") == []
    assert items[0]["line_total"] == 30775


def test_missing_line_total_is_never_filled_in():
    items = [{"quantity": 475, "unit_price": 65.0, "line_total": None}]
    assert llm_extract.apply_line_item_guardrails(items, _PRINTED) == []
    assert items[0]["line_total"] is None


def test_consistent_line_total_is_left_alone():
    items = [{"quantity": 475, "unit_price": 65.0, "line_total": 30875.0}]
    assert llm_extract.apply_line_item_guardrails(items, _PRINTED) == []
    assert items[0]["line_total"] == 30875.0


def test_line_guardrail_tolerates_junk_input():
    assert llm_extract.apply_line_item_guardrails(None, "x") == []
    assert llm_extract.apply_line_item_guardrails(["not a dict", {"quantity": "n/a"}], "x") == []


# ── Hetero invoice numbers: "SI" + 10 digits, undoing OCR prefix misreads ─
# Every spelling below was seen in the real Hetero batch's OCR text.

def _hetero(number, text, supplier="HETERO LABS LIMITED"):
    header = {"supplier_name": supplier, "invoice_number": number}
    notes = []
    llm_extract.apply_header_guardrails(header, [], text, notes=notes)
    return header["invoice_number"], notes


def test_hetero_prefix_misreads_are_repaired_without_touching_digits():
    for garbled, text_digits in [
        ("S13626204505", "3626204505"),     # letter I read as digit 1 (132622, all pages)
        ("SI13626102057", "3626102057"),    # extra stray 1 (57_...1 page 1)
        ("813626102057", "3626102057"),      # S read as 8 (57_...1 pages 15, 17)
        ("513626204505", "3626204505"),      # S read as 5 (132622 page 3, one OCR run)
        ("SI13626102159", "3626102159"),
    ]:
        fixed, notes = _hetero(garbled, f"Invoice No. {garbled} Dt:22.08.2026")
        assert fixed == "SI" + text_digits, garbled
        assert notes and garbled in notes[0] and "digits are unchanged" in notes[0]


def test_hetero_correct_number_is_left_alone():
    fixed, notes = _hetero("SI3626204603", "SI3626204603 Dt:26.08.2026")
    assert fixed == "SI3626204603" and notes == []


def test_hetero_repair_needs_the_digits_to_be_in_the_text():
    # Nothing in the document backs the 10-digit core -> nothing is repaired.
    fixed, notes = _hetero("S13626204505", "an unrelated page of text")
    assert fixed == "S13626204505" and notes == []


def test_hetero_repair_never_changes_a_digit():
    fixed, _ = _hetero("S13626204505", "S13626204505")
    assert fixed[2:] == "3626204505"


def test_format_rule_applies_only_to_hetero():
    fixed, notes = _hetero("S13626204505", "S13626204505", supplier="ACME EXPORTS PVT LTD")
    assert fixed == "S13626204505" and notes == []
    fixed, notes = _hetero("S13626204505", "S13626204505", supplier="")
    assert fixed == "S13626204505" and notes == []


def test_hetero_numbers_of_unexpected_shape_are_not_guessed_at():
    for odd in ("S1362620450", "SI36262045051234", "INV-2026-001", "S1ABC6204505"):
        fixed, notes = _hetero(odd, odd)
        assert fixed == odd and notes == [], odd


def test_hetero_collapse_then_repair_end_to_end():
    # 132622-style: duplicates by OCR variant AND a garbled prefix.
    header = {
        "supplier_name": "HETERO LABS LIMITED",
        "invoice_number": "S13626204603 / 513626204603",
        "invoice_date": "2026-08-26 / 2026-08-26", "invoice_total": "30875.00 / 30875.00",
    }
    notes = []
    llm_extract.apply_header_guardrails(header, [], "S13626204603 and 513626204603", notes=notes)
    assert header["invoice_number"] == "SI3626204603"
    assert header["invoice_total"] == "30875.00"
    assert any("collapsed" in n for n in notes) and any("repaired" in n for n in notes)


def test_each_hetero_number_in_a_combined_list_is_repaired_separately():
    fixed, notes = _hetero("S13626204603 / SI3726201956", "S13626204603 SI3726201956")
    assert fixed == "SI3626204603 / SI3726201956" and len(notes) == 1


# ── HSN code: the Item No column must not leak into it (real JCB rows) ───

_JCB_ROWS = """\
SNo   Item │HSN  Code │ Part No   │   Description   │   Qty │UM │ Ship-to │ Basic │ Total
   No   │   Party's │ Weight
   1  │  160 │8481 80 90  │ 402/Y3499 MANUAL EVB W/O KPC W/O SE│2  │  NOS  │  4500573893  │  1,750.29  │  3,500.58
   2  │  150 │8431 49 90  │ 333/K2338│ABI-HEADER TANK JS145 │ 2 │ NOS │ 4500560852 │ 98.67 │ 197.34
   3  │  151 │4010 35 90  │ 320/08600│BELT.FRONT8PKtbaAR3#VM117 │ 1 │ NOS │ 4500565332 │ 29.99 │ 29.99
"""


def _hsn_item(hsn, part="", desc=""):
    return {"hsn_code": hsn, "part_number": part, "product_description": desc}


def test_item_number_in_hsn_is_replaced_by_the_hsn_printed_on_that_row():
    # Real case: Item No 160 sat in the HSN field; the page prints 8481 80 90.
    items = [_hsn_item("160", part="402/Y3499")]
    notes = llm_extract.apply_line_item_guardrails(items, _JCB_ROWS)
    assert items[0]["hsn_code"] == "8481 80 90"
    assert notes and "'160'" in notes[0] and "8481 80 90" in notes[0]


def test_row_is_found_by_description_when_there_is_no_part_number():
    items = [_hsn_item("160", desc="MANUAL EVB W/O KPC W/O SE")]
    llm_extract.apply_line_item_guardrails(items, _JCB_ROWS)
    assert items[0]["hsn_code"] == "8481 80 90"


def test_each_item_takes_the_hsn_from_its_own_row():
    items = [_hsn_item("150", part="333/K2338"), _hsn_item("151", part="320/08600")]
    llm_extract.apply_line_item_guardrails(items, _JCB_ROWS)
    assert [it["hsn_code"] for it in items] == ["8431 49 90", "4010 35 90"]


def test_item_number_fused_onto_the_hsn_by_ocr_is_stripped():
    # Real scanned rows: "420 4016 93 40" and "480 84314980".
    text = "1 │ 420 4016 93 40 │ 332/Y3300 GASKET VALVE COVER\n3 │ 480 84314980 │ 400/F0345 TOOTH\n"
    items = [_hsn_item("420 4016 93 40"), _hsn_item("480 84314980")]
    notes = llm_extract.apply_line_item_guardrails(items, text)
    assert [it["hsn_code"] for it in items] == ["4016 93 40", "84314980"]
    assert len(notes) == 2 and "fused" in notes[0]


def test_a_valid_hsn_is_never_touched():
    for ok in ("8431 49 90", "84314980", "4016 93 40", "8431"):
        items = [_hsn_item(ok, part="333/K2338")]
        assert llm_extract.apply_line_item_guardrails(items, _JCB_ROWS) == []
        assert items[0]["hsn_code"] == ok


def test_fused_value_is_left_alone_when_the_hsn_is_not_in_the_source():
    items = [_hsn_item("420 4016 93 40")]
    assert llm_extract.apply_line_item_guardrails(items, "nothing relevant here") == []
    assert items[0]["hsn_code"] == "420 4016 93 40"


def test_short_value_is_left_alone_when_the_row_has_no_hsn_or_disagrees():
    items = [_hsn_item("160", part="402/Y3499")]
    no_hsn = "1 │ 160 │ │ 402/Y3499 MANUAL EVB │ 2 │ NOS\n"
    assert llm_extract.apply_line_item_guardrails(items, no_hsn) == [] and items[0]["hsn_code"] == "160"
    # same part printed twice with DIFFERENT HSNs -> ambiguous, don't pick one
    two = "1 │ 160 │ 8481 80 90 │ 402/Y3499 A\n2 │ 170 │ 7318 15 00 │ 402/Y3499 A\n"
    items = [_hsn_item("160", part="402/Y3499")]
    assert llm_extract.apply_line_item_guardrails(items, two) == [] and items[0]["hsn_code"] == "160"


def test_blank_or_missing_hsn_is_never_filled_in():
    for blank in ("", None):
        items = [_hsn_item(blank, part="402/Y3499")]
        assert llm_extract.apply_line_item_guardrails(items, _JCB_ROWS) == []
        assert items[0]["hsn_code"] == blank


# ── item-table header context for continuation pages ────────────────────

def test_table_header_context_returns_the_header_lines_above_the_first_row():
    page = "Exporter ...\n" + _JCB_ROWS + "Total │ 5 │ 3,727.92\n"
    ctx = llm_extract._table_header_context(page)
    assert ctx.splitlines()[0].startswith("SNo") and "Party's" in ctx
    assert "402/Y3499" not in ctx  # stops before the first data row


def test_table_header_context_is_empty_without_a_header():
    assert llm_extract._table_header_context("just some text\nwith no table\n") == ""


# ── fine_chunks: one call per page, header carried to continuation pages ─

def test_fine_chunks_makes_one_call_per_page_with_the_header_on_later_pages(monkeypatch):
    calls = []

    async def fake_extract_line_items(text, **kwargs):
        calls.append((text, kwargs.get("feedback")))
        return [{"item_ser_no": len(calls)}], dict(_USAGE)

    monkeypatch.setattr(llm_extract, "extract_line_items", fake_extract_line_items)
    text = _page_marker(1, 2) + _JCB_ROWS + _page_marker(2, 2) + "15 │ 350 │ 8483 90 00 │ 332/Y8134 PLATE │ 12 │ NOS\n"

    items, usage = asyncio.run(llm_extract.extract_line_items_chunked(
        text, model="gpt-5-nano", feedback="your total is short", fine_chunks=True,
    ))

    assert len(calls) == 2  # a 2-page document is normally a SINGLE call
    assert "PAGE 1 OF 2" in calls[0][0] and "COLUMN HEADERS" not in calls[0][0]
    assert "COLUMN HEADERS OF THE ITEM TABLE" in calls[1][0] and "SNo" in calls[1][0]
    assert "15 │ 350" in calls[1][0]
    assert [c[1] for c in calls] == [None, None]  # a per-page call can't act on a document-wide complaint
    assert len(items) == 2 and usage["input_tokens"] == _USAGE["input_tokens"] * 2


def test_fine_chunks_on_a_single_page_document_is_a_single_normal_call(monkeypatch):
    calls = []

    async def fake_extract_line_items(text, **kwargs):
        calls.append(kwargs)
        return [], dict(_USAGE)

    monkeypatch.setattr(llm_extract, "extract_line_items", fake_extract_line_items)
    asyncio.run(llm_extract.extract_line_items_chunked(
        _page_marker(1, 1) + "only page\n", model="gpt-5-nano", feedback="fb", fine_chunks=True,
    ))
    assert len(calls) == 1 and calls[0]["feedback"] == "fb"


# ── unit price taken for the line amount (real JCB MH2706215472 row) ────

_BELL_ROW = """\
   1  │  160 │8481 80 90  │ 402/Y3499 MANUAL EVB W/O KPC W/O SE│2  │  NOS  │  4500573893   │   40.000 (KG)│80.000   │   IN   │   1,750.29  │  3,500.58 │ 3,500.58
   Total   │   2   │   3,500.58 │ 3,500.58
"""


def _shift_item(qty=2, price=875.145, total=1750.29, part="402/Y3499"):
    return {"quantity": qty, "unit_price": price, "line_total": total, "part_number": part,
            "product_description": "MANUAL EVB W/O KPC W/O SE", "hsn_code": "8481 80 90"}


def test_unit_price_taken_for_the_line_amount_is_corrected_from_the_printed_row():
    # The model returned price 875.145 (printed nowhere) and total 1750.29
    # (really the unit price). The row prints 1,750.29 and 3,500.58.
    items = [_shift_item()]
    notes = llm_extract.apply_line_item_guardrails(items, _BELL_ROW)
    assert items[0]["unit_price"] == 1750.29 and items[0]["line_total"] == 3500.58
    assert notes and "875.145" in notes[0] and "3500.58" in notes[0]


def test_shift_repair_works_when_the_row_is_found_by_description_alone():
    items = [_shift_item(part="")]
    llm_extract.apply_line_item_guardrails(items, _BELL_ROW)
    assert items[0]["line_total"] == 3500.58


def test_a_correct_row_is_never_touched():
    items = [_shift_item(price=1750.29, total=3500.58)]
    assert llm_extract.apply_line_item_guardrails(items, _BELL_ROW) == []
    assert (items[0]["unit_price"], items[0]["line_total"]) == (1750.29, 3500.58)


def test_a_printed_unit_price_is_never_second_guessed():
    # price 98.67 IS printed on the row -> a real price, however odd the rest.
    row = "1 │ 150 │ 8431 49 90 │ 333/K2338 ABI-HEADER TANK │ 2 │ NOS │ 98.67 │ 197.34 │ 394.68\n"
    items = [{"quantity": 2, "unit_price": 98.67, "line_total": 197.34, "part_number": "333/K2338", "product_description": "ABI-HEADER TANK"}]
    assert llm_extract.apply_line_item_guardrails(items, row) == []
    assert items[0]["line_total"] == 197.34


def test_quantity_one_rows_are_not_shift_candidates():
    row = "1 │ 150 │ 8431 49 90 │ 333/K2338 ABI-HEADER TANK │ 1 │ NOS │ 98.67 │ 98.67\n"
    items = [{"quantity": 1, "unit_price": 98.67, "line_total": 98.67, "part_number": "333/K2338", "product_description": "ABI-HEADER TANK"}]
    assert llm_extract.apply_line_item_guardrails(items, row) == []


def test_shift_repair_needs_the_quantity_multiple_to_be_printed_on_the_row():
    row = "1 │ 160 │ 8481 80 90 │ 402/Y3499 MANUAL EVB │ 2 │ NOS │ 1,750.29\n"  # no 3,500.58 anywhere
    items = [_shift_item()]
    assert llm_extract.apply_line_item_guardrails(items, row) == []
    assert (items[0]["unit_price"], items[0]["line_total"]) == (875.145, 1750.29)


def test_shift_repair_is_skipped_when_the_item_row_cannot_be_found():
    items = [_shift_item(part="NOPE-123")]
    items[0]["product_description"] = "SOMETHING ELSE ENTIRELY"
    assert llm_extract.apply_line_item_guardrails(items, _BELL_ROW) == []


# ── wrong-column prices/amounts, repaired from the printed row ──────────
# Real misreads from live JCB runs (rows as the source prints them).

_ROW_4 = "4 │ 130 │ 8512 90 00 │ 402/T3156│WA REAR COMBI LAMP GUARD │ 3 │ NOS │ 4500576303 1010983380 │ 1.500 (KG) │ 4.500 │ IN │ 17.44 │ 52.32 │ 52.32\n"
_ROW_9 = "9 │ 140 │ 8482 40 00 │ 917/02400│NEEDLE ROLLER BRG │ 2 │ NOS │ 4500576303 1010983380 │ 0.070 (KG) │ 0.140 │ SK │ 30.32 │ 60.64 │ 60.64\n"
_ROW_22 = "22 │ 330 │ 7007 11 00 │ 334/Y0833│UPPER DOOR TINTED GLASS-RH │ 2 │ NOS │ 10182 │ 10.800 (KG) │ 21.600 │ IN │ 24.46 │ 48.92 │ 48.92\n"


def _wrong(part, desc, qty, price, total):
    return {"part_number": part, "product_description": desc, "quantity": qty, "unit_price": price, "line_total": total}


def test_total_weight_taken_as_the_unit_price_is_corrected():
    # 220038 row 4: price 4.5 is the TOTAL WEIGHT, total 13.5 derived from it.
    items = [_wrong("402/T3156", "WA REAR COMBI LAMP GUARD", 3.0, 4.5, 13.5)]
    llm_extract.apply_line_item_guardrails(items, _ROW_4)
    assert (items[0]["unit_price"], items[0]["line_total"]) == (17.44, 52.32)


def test_an_invented_price_and_total_are_corrected():
    # 220038 row 9: 0.38 / 0.76 appear nowhere on the row.
    items = [_wrong("917/02400", "NEEDLE ROLLER BRG", 2.0, 0.38, 0.76)]
    llm_extract.apply_line_item_guardrails(items, _ROW_9)
    assert (items[0]["unit_price"], items[0]["line_total"]) == (30.32, 60.64)


def test_basic_value_taken_as_the_unit_price_is_corrected():
    # 214961 row 22: 48.92 is the Basic Value; 97.84 was derived from it.
    items = [_wrong("334/Y0833", "UPPER DOOR TINTED GLASS-RH", 2.0, 48.92, 97.84)]
    notes = llm_extract.apply_line_item_guardrails(items, _ROW_22)
    assert (items[0]["unit_price"], items[0]["line_total"]) == (24.46, 48.92)
    assert notes and "24.46" in notes[0] and "Basic Value and Total" in notes[0]


def test_weight_columns_never_win_over_the_money_pair():
    # 10.800 x 2 = 21.600 is also a qty-multiple pair on the row, but a
    # weight prints once; only the twice-printed amount qualifies.
    items = [_wrong("334/Y0833", "UPPER DOOR TINTED GLASS-RH", 2.0, 10.8, 97.84)]  # a weight as the price, total derived
    llm_extract.apply_line_item_guardrails(items, _ROW_22)
    assert (items[0]["unit_price"], items[0]["line_total"]) == (24.46, 48.92)


def test_a_weight_pair_taken_as_price_and_amount_is_corrected_to_the_money_pair():
    # 10.800 / 21.600 are WEIGHTS (three decimals), not money, so they don't
    # count as "printed" -- the money pair on the row wins.
    items = [_wrong("334/Y0833", "UPPER DOOR TINTED GLASS-RH", 2.0, 10.8, 21.6)]
    llm_extract.apply_line_item_guardrails(items, _ROW_22)
    assert (items[0]["unit_price"], items[0]["line_total"]) == (24.46, 48.92)


def test_two_competing_printed_pairs_are_never_guessed_between():
    row = "1 │ 1 │ 8481 80 90 │ 402/Y3499 THING │ 2 │ NOS │ 5.00 │ 10.00 │ 10.00 │ 7.00 │ 14.00 │ 14.00\n"
    items = [_wrong("402/Y3499", "THING", 2.0, 99.0, 198.0)]
    assert llm_extract.apply_line_item_guardrails(items, row) == []
    assert (items[0]["unit_price"], items[0]["line_total"]) == (99.0, 198.0)


def test_a_row_where_price_and_amount_are_both_printed_is_left_alone():
    items = [_wrong("402/T3156", "WA REAR COMBI LAMP GUARD", 3.0, 17.44, 52.32)]
    assert llm_extract.apply_line_item_guardrails(items, _ROW_4) == []


# ── scanned rows: decimal commas and amounts wrapped onto the next line ──
# The REAL OCR text of the scanned JCB invoice MH2706184170 (both scans).

_SCAN = """\
   1 │ 420 4016 93 40 │ 332/Y3300 | GASKET VALVE COVER │ 12 │ NOS │ 29126 1010757421 │ 0.026 (KG) │ 0 ’ 312 │ IN 2,00
   24,00 │ 24,00
   2 │ 450 8511 50 00 │ 320/08680 | ALTERNATOR │ J │ NOS │ 29126 1010757421 │ 5.210 (KG) │ 5 ’ 210 │ IN 149,34
   149,34 │ 149,34
   3 │ 480 84314980 │ 400/F0345 | JCB Supertooth - RHS Side Cutter│10│NOS │ 29126| 1010757421 │ 4.870 (KG) │ 48,700 │ CN
   22,65 226,50 │ 226,50
   4 │ 630 |84314990 │ 400/F0345 | JCB Supertooth - RHS Side Cutter│12│|NOS │ 29126/ 1010757421 │ 4.870 (KG) │ 58,440 │ CN
   22,65 271,80 │ 271,80
   5 │ 640 4016 93 40 │ 332/Y3300 | GASKET VALVE COVER │ 6 │ NOS │ 29126| 1010757421 │ 0.026 (KG) │ 0 ,156 │ IN
   2,00 12,00 │ 12,00
   6 │ 641 8409 99 13 │ 320/03388 | PISTON RING.TOP │ 40 │ NOS │ 29126| 1010757380 │ 0.186 (KG) │ 7 ’ 440 │ IN 534
   213,60 │ 213,60
   7 │ 790 | 8511 50 00 │ 320/08680 | ALTERNATOR │ 2 │ NOS │ 29126/ 1010757380 │ 5.210 (KG) │ 10,420 │ IN 149,34 298,68
   298,68
   8 │ 791 8482 20 90 │ 907/08300 | TAPER ROLLER BEARING │ 15 │ NOS │ 29126| 1010757359 │ 0.626 (KG) │ 9 ’ 3590 │ us 20,95 314,25 │ 314,25
   9 │ 792 | 84314930 │ 402/P4714 | WA BOTTOM PIN KIP SIS │ 8 │ NOS │ 29126 1010758404 │ 0.001 (KG) │ 0,008 │ L 36,63 293,04 │ 293,04
   Total │ 106
   1.803,21 │ 1.803,21
"""


def test_money_amounts_are_read_in_both_decimal_conventions_and_weights_are_not():
    from collections import Counter
    from decimal import Decimal as D
    c = llm_extract._money_counts("1,750.29  22,65  1.803,21  3,500.58 3,500.58  48,700  5.210  29.07.2026  4500573893")
    assert c == Counter({D("1750.29"): 1, D("22.65"): 1, D("1803.21"): 1, D("3500.58"): 2})


def test_amount_wrapped_onto_the_next_line_is_found_and_used():
    # Row 3: the model returned price 226.5 / total 2265 (derived). The row
    # prints "22,65 226,50 | 226,50" on the line UNDER it.
    items = [_wrong("400/F0345", "JCB Supertooth - RHS Side Cutter", 10, 226.5, 2265.0)]
    llm_extract.apply_line_item_guardrails(items, _SCAN)
    assert (items[0]["unit_price"], items[0]["line_total"]) == (22.65, 226.5)


def test_row_4_and_row_9_of_the_scan_are_corrected_too():
    items = [
        _wrong("400/F0345", "JCB Supertooth - RHS Side Cutter", 12, 271.8, 3261.6),   # price/amount derived from the amount
        _wrong("402/P4714", "WA BOTTOM PIN KIP SIS", 8, 293.04, 293.04),              # both printed, but 8 x 293.04 != 293.04
    ]
    llm_extract.apply_line_item_guardrails(items, _SCAN)
    assert (items[0]["unit_price"], items[0]["line_total"]) == (22.65, 271.8)
    assert (items[1]["unit_price"], items[1]["line_total"]) == (36.63, 293.04)


def test_correct_scan_rows_are_left_alone():
    items = [
        _wrong("332/Y3300", "GASKET VALVE COVER", 12, 2.0, 24.0),
        _wrong("320/08680", "ALTERNATOR", 2, 149.34, 298.68),
        _wrong("907/08300", "TAPER ROLLER BEARING", 15, 20.95, 314.25),
    ]
    assert llm_extract.apply_line_item_guardrails(items, _SCAN) == []


def test_a_price_ocr_printed_without_its_decimal_is_not_guessed_at():
    # Row 6 prints "534" for 5.34 (OCR dropped the comma): the price can't
    # be found as money, so nothing is "corrected" -- the model's 5.34 stays.
    items = [_wrong("320/03388", "PISTON RING.TOP", 40, 5.34, 213.6)]
    assert llm_extract.apply_line_item_guardrails(items, _SCAN) == []
    assert (items[0]["unit_price"], items[0]["line_total"]) == (5.34, 213.6)


def test_a_continuation_line_never_borrows_the_next_rows_numbers():
    # Item A's own row prints no amounts; the NEXT line starts row B and
    # prints 7.00 / 14.00 twice. Those belong to B and must not be used.
    text = (
        "1 │ 10 │ 8481 80 90 │ AAA-1111 WIDGET ASSEMBLY │ 2 │ NOS\n"
        "2 │ 20 │ 8481 80 90 │ BBB-2222 GADGET ASSEMBLY │ 2 │ NOS │ 7.00 14.00 │ 14.00\n"
    )
    items = [_wrong("AAA-1111", "WIDGET ASSEMBLY", 2, 99.0, 198.0)]
    assert llm_extract.apply_line_item_guardrails(items, text) == []
    assert (items[0]["unit_price"], items[0]["line_total"]) == (99.0, 198.0)


def test_the_hsn_lookup_does_not_use_continuation_lines():
    # The line under item A is row B, with B's HSN. A (no HSN on its own
    # row) must not be handed B's code.
    text = (
        "1 │ 10 │ │ AAA-1111 WIDGET ASSEMBLY │ 2 │ NOS\n"
        "2 │ 20 │ 7318 15 00 │ BBB-2222 GADGET ASSEMBLY │ 2 │ NOS\n"
    )
    items = [{"hsn_code": "10", "part_number": "AAA-1111", "product_description": "WIDGET ASSEMBLY"}]
    assert llm_extract.apply_line_item_guardrails(items, text) == []
    assert items[0]["hsn_code"] == "10"


def test_price_printed_without_its_decimal_separator_is_recovered_when_the_arithmetic_closes():
    # Scan row 6 prints "534" for 5.34 and the amount "213,60" twice; 40 x
    # 5.34 = 213.60 exactly. The model returned (213.6, 8544.0) / (0.186, 7.44).
    for bad in [(213.6, 8544.0), (0.186, 7.44)]:
        items = [_wrong("320/03388", "PISTON RING.TOP", 40, *bad)]
        notes = llm_extract.apply_line_item_guardrails(items, _SCAN)
        assert (items[0]["unit_price"], items[0]["line_total"]) == (5.34, 213.6), bad
        assert notes


def test_a_bare_integer_is_not_taken_as_a_price_without_a_matching_amount():
    # "150" appears (an item number) but no amount printed twice equals
    # quantity x 1.50 -> no pair, nothing changes.
    text = "1 │ 150 │ 8481 80 90 │ AAA-1111 WIDGET ASSEMBLY │ 2 │ NOS │ 9.99\n"
    items = [_wrong("AAA-1111", "WIDGET ASSEMBLY", 2, 1.5, 3.0)]
    assert llm_extract.apply_line_item_guardrails(items, text) == []
    assert (items[0]["unit_price"], items[0]["line_total"]) == (1.5, 3.0)


def test_the_whole_scanned_invoice_reconciles_to_its_printed_total():
    # Every row, from the model's worst real output, repaired from the page;
    # the nine amounts must add up to the printed 1.803,21.
    wrong = [
        ("332/Y3300", "GASKET VALVE COVER", 12, 2.0, 24.0), ("320/08680", "ALTERNATOR", 1, 149.34, 149.34),
        ("400/F0345", "JCB Supertooth - RHS Side Cutter", 10, 4.87, 48.7), ("400/F0345", "JCB Supertooth - RHS Side Cutter", 12, 4.87, 58.44),
        ("332/Y3300", "GASKET VALVE COVER", 6, 0.026, 0.156), ("320/03388", "PISTON RING.TOP", 40, 0.186, 7.44),
        ("320/08680", "ALTERNATOR", 2, 5.21, 10.42), ("907/08300", "TAPER ROLLER BEARING", 15, 0.626, 9.359),
        ("402/P4714", "WA BOTTOM PIN KIP SIS", 8, 0.001, 0.008),
    ]
    items = [_wrong(*w) for w in wrong]
    llm_extract.apply_line_item_guardrails(items, _SCAN)
    assert round(sum(i["line_total"] for i in items), 2) == 1803.21


# ── quantity-1 rows where the weight columns were taken as price/amount ──

_ROW_QTY1 = "16 │ 880 │ 9032 89 00 │ 400/25617│DVS Electrohydraulic cont │ 1 │ NOS │ 4500578245 │ 550.000 (KG) │550.000 │ IN │ 173.01 │ 173.01 │ 173.01\n"


def test_weights_taken_as_price_and_amount_on_a_quantity_one_row_are_corrected():
    # Real 227647 row 16: price and amount 550.0 are the net/total WEIGHT
    # columns; the row prints 173.01 three times (price, Basic Value, Total).
    items = [_wrong("400/25617", "DVS Electrohydraulic cont", 1, 550.0, 550.0)]
    notes = llm_extract.apply_line_item_guardrails(items, _ROW_QTY1)
    assert (items[0]["unit_price"], items[0]["line_total"]) == (173.01, 173.01)
    assert notes


def test_a_correct_quantity_one_row_is_left_alone():
    items = [_wrong("400/25617", "DVS Electrohydraulic cont", 1, 173.01, 173.01)]
    assert llm_extract.apply_line_item_guardrails(items, _ROW_QTY1) == []


def test_quantity_one_row_with_an_amount_printed_only_once_is_not_touched():
    row = "1 │ 10 │ 8481 80 90 │ AAA-1111 WIDGET ASSEMBLY │ 1 │ NOS │ 550.000 (KG) │ 173.01\n"
    items = [_wrong("AAA-1111", "WIDGET ASSEMBLY", 1, 550.0, 550.0)]
    assert llm_extract.apply_line_item_guardrails(items, row) == []
    assert (items[0]["unit_price"], items[0]["line_total"]) == (550.0, 550.0)


def test_quantity_one_row_with_two_different_repeated_amounts_is_not_guessed():
    row = "1 │ 10 │ 8481 80 90 │ AAA-1111 WIDGET ASSEMBLY │ 1 │ NOS │ 173.01 │ 173.01 │ 12.50 │ 12.50\n"
    items = [_wrong("AAA-1111", "WIDGET ASSEMBLY", 1, 550.0, 550.0)]
    assert llm_extract.apply_line_item_guardrails(items, row) == []


# ── freight / charge lines are not line items ───────────────────────────

_FREIGHT_TEXT = (
    "1391539 │ LON-CM-12EC-020U_X4   PSC Board   1   INR   22,476.03   INR   22,476.03\n"
    "FREIGHT:   INR   674.28\n"
    "Total Value INR   23,150.31\n"
)


def test_freight_printed_once_under_the_table_is_removed_even_with_a_misread_quantity():
    # Real case: "6 74.28" was split by the text layer and read as 6 x 74.28.
    items = [
        {"product_description": "PSC Board", "quantity": 1, "unit_price": 22476.03, "line_total": 22476.03},
        {"product_description": "FREIGHT", "quantity": 6, "unit_price": 74.28, "line_total": 445.68},
    ]
    notes = llm_extract.apply_line_item_guardrails(items, _FREIGHT_TEXT)
    assert [i["product_description"] for i in items] == ["PSC Board"]
    assert any("Freight" in n for n in notes)


def test_freight_row_with_no_quantity_is_removed():
    items = [
        {"product_description": "PSC Board", "quantity": 1, "unit_price": 22476.03, "line_total": 22476.03},
        {"product_description": "Freight:", "quantity": None, "unit_price": None, "line_total": 674.28},
    ]
    llm_extract.apply_line_item_guardrails(items, _FREIGHT_TEXT)
    assert len(items) == 1


def test_freight_row_is_kept_when_the_source_has_no_single_amount_freight_line():
    # A real priced table row called "Freight" (several amounts on its line)
    # is a genuine line, not a trailing charge.
    text = "9   Freight   1   LOT   500.00   500.00\n"
    items = [{"product_description": "Freight", "quantity": 1, "unit_price": 500.0, "line_total": 500.0}]
    llm_extract.apply_line_item_guardrails(items, text)
    assert len(items) == 1


def test_product_whose_name_merely_contains_freight_is_kept():
    items = [{"product_description": "Freight container lock", "quantity": 2, "unit_price": 5.0, "line_total": 10.0}]
    llm_extract.apply_line_item_guardrails(items, "FREIGHT: INR 3.50\n")
    assert len(items) == 1


# ── air waybill pages are not part of the item table ────────────────────

_AWB_PAGE = (
    "=== PAGE 1 OF 3 (x.pdf) ===\n"
    "Shipper's Name and Address   │   Shipper's Account Number   Not Negotiable\n"
    "Air Waybill   Issued by\nConsignee's Name and Address\n"
    "Nature and Quantity of Goods   Computer Equipment-454896\n"
)
_INVOICE_PAGE = "=== PAGE 2 OF 3 (x.pdf) ===\nCommercial Invoice: 454896\n1031580 │ QSFP28 CABLE │ 2 │ INR 8,123.66\n"
_PACKING_PAGE = "=== PAGE 3 OF 3 (x.pdf) ===\nPACKING LIST\n"


def test_air_waybill_page_is_dropped_from_the_line_items_text():
    out = llm_extract._drop_transport_document_pages(_AWB_PAGE + _INVOICE_PAGE + _PACKING_PAGE)
    assert "Computer Equipment" not in out
    assert "QSFP28 CABLE" in out and "PACKING LIST" in out


def test_a_document_that_is_only_a_waybill_is_left_alone():
    assert llm_extract._drop_transport_document_pages(_AWB_PAGE) == _AWB_PAGE


def test_invoice_page_that_merely_mentions_a_waybill_is_kept():
    page = "=== PAGE 1 OF 2 (x.pdf) ===\nInvoice. Air Waybill No. 123-4567\n1 │ WIDGET │ 2 │ 5.00\n"
    text = page + _PACKING_PAGE.replace("PAGE 3 OF 3", "PAGE 2 OF 2")
    assert llm_extract._drop_transport_document_pages(text) == text


_HTS_ROW = "1002760 │ AIP-NDAAFF0006 NDAAFF0006 QSFP28 Cable CN │ 8517.62.0000   8517.62.90   1   INR   2,454.08  INR   2,454.08\n"


def test_ten_digit_tariff_code_is_replaced_by_the_india_hts_printed_on_the_same_line():
    items = [{"product_description": "QSFP28 Cable", "part_number": "AIP-NDAAFF0006", "hsn_code": "8517.62.0000",
              "quantity": 1, "unit_price": 2454.08, "line_total": 2454.08}]
    notes = llm_extract.apply_line_item_guardrails(items, _HTS_ROW)
    assert items[0]["hsn_code"] == "8517.62.90"
    assert notes


def test_ten_digit_code_with_no_matching_india_code_is_left_alone():
    items = [{"product_description": "X", "part_number": "P", "hsn_code": "8517.62.0000",
              "quantity": 1, "unit_price": 5.0, "line_total": 5.0}]
    assert llm_extract.apply_line_item_guardrails(items, "P │ 8517.62.0000 │ 1 │ 5.00 │ 5.00\n") == []
    assert items[0]["hsn_code"] == "8517.62.0000"


# ── a repeated part number must not pool several rows' numbers ──────────

_REPEATED_PART_TEXT = (
    "445-0788888  BRACKET - SR SLIDE │84734090│1010011111│ 3310000001 │  24.000  │  3.6600  │   87.84\n"
    "445-0788888  BRACKET - SR SLIDE │84734090│1010011111│ 3310000002 │ 100.000  │  6.4800  │  648.00\n"
    "445-0799999  EPP BRACKET ASSY   │84734090│1010011112│ 3310000003 │  96.000  │  6.7500  │  648.00\n"
)


def test_a_correct_row_is_not_repaired_from_another_row_with_the_same_part_number():
    # Real Nash case: the same part prints on several rows (different POs) and
    # another row repeats the amount 648.00, so pooled numbers looked like one
    # row's "Basic Value + Total" and a CORRECT 24 x 3.66 = 87.84 was
    # overwritten with 648.00.
    items = [{"product_description": "BRACKET - SR SLIDE", "part_number": "445-0788888", "quantity": 24,
              "unit_price": 3.66, "line_total": 87.84}]
    assert llm_extract.apply_line_item_guardrails(items, _REPEATED_PART_TEXT) == []
    assert (items[0]["unit_price"], items[0]["line_total"]) == (3.66, 87.84)


def test_continuation_never_runs_into_the_next_rows_numbers():
    text = (
        "445-0700001  WIDGET A │84734090│1010011111│ 3310000001 │  5.000  │  1.0000  │   5.00\n"
        "445-0700002  WIDGET B │84734090│1010011112│ 3310000002 │  4.000  │  162.0000 │  648.00\n"
        "445-0700003  WIDGET C │84734090│1010011113│ 3310000003 │  4.000  │  162.0000 │  648.00\n"
    )
    blocks = llm_extract._source_row_blocks({"part_number": "445-0700001"}, text)
    assert len(blocks) == 1 and "648.00" not in blocks[0]
