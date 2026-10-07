"""Smoke tests for validate.py's coercion, mapping, and no-confidence output."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import load_template_schema
from validate import (
    _clean_generic_item,
    _clean_invoice_header,
    _coerce_number,
    _map_export_item,
    _map_import_item,
    _renumber_items_if_ser_no_resets,
    validate_and_coerce,
)

_TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def test_currency_string_coercion():
    assert _coerce_number("$1,234.50") == 1234.50
    assert _coerce_number("₹1,000") == 1000
    assert _coerce_number("not a number") is None
    assert _coerce_number(42) == 42


def test_map_export_item_basic_fields():
    item = {
        "item_ser_no": 1, "product_description": "ROTOR SHAFT", "part_number": "",
        "model_or_type": "", "order_number": "", "hsn_code": "84831099",
        "quantity": 10, "unit_of_measurement": "PCS", "unit_price": 5.0,
        "line_total": 50.0, "end_use_code": "", "fta_code": "",
        "country_of_origin": "", "igst_amount": "", "igst_rate": "", "packages": "",
    }
    mapped = _map_export_item(item, {})
    assert mapped["Item_RITC"] == "84831099"
    assert mapped["ItmTaxableVal"] == 50.0


def test_map_export_item_uses_own_packages_over_header_total():
    # Confirmed real case: two items on the same invoice have their OWN,
    # DIFFERENT package counts (108 and 504) -- each must keep its own
    # number, not the header's whole-shipment total copied onto both.
    item = {
        "item_ser_no": 1, "product_description": "ROTOR SHAFT", "part_number": "",
        "model_or_type": "", "order_number": "", "hsn_code": "84831099",
        "quantity": 10, "unit_of_measurement": "PCS", "unit_price": 5.0,
        "line_total": 50.0, "end_use_code": "", "fta_code": "",
        "country_of_origin": "", "igst_amount": "", "igst_rate": "", "packages": 108,
    }
    mapped = _map_export_item(item, {"total_packages": 612})
    assert mapped["ItmTotpkg"] == 108


def test_map_export_item_falls_back_to_header_packages_when_item_has_none():
    item = {
        "item_ser_no": 1, "product_description": "ROTOR SHAFT", "part_number": "",
        "model_or_type": "", "order_number": "", "hsn_code": "84831099",
        "quantity": 10, "unit_of_measurement": "PCS", "unit_price": 5.0,
        "line_total": 50.0, "end_use_code": "", "fta_code": "",
        "country_of_origin": "", "igst_amount": "", "igst_rate": "", "packages": "",
    }
    mapped = _map_export_item(item, {"total_packages": 6})
    assert mapped["ItmTotpkg"] == 6


def test_clean_generic_item_coerces_packages_field():
    raw_item = {
        "item_ser_no": 1, "product_description": "ROTOR SHAFT",
        "quantity": 10, "unit_price": 5.0, "line_total": 50.0,
        "packages": 108,
    }
    cleaned = _clean_generic_item(raw_item, warnings=[], row_idx=0, ser_no=1)
    assert cleaned["packages"] == 108


def test_clean_generic_item_defaults_missing_packages_to_empty_string():
    raw_item = {
        "item_ser_no": 1, "product_description": "ROTOR SHAFT",
        "quantity": 10, "unit_price": 5.0, "line_total": 50.0,
    }
    cleaned = _clean_generic_item(raw_item, warnings=[], row_idx=0, ser_no=1)
    assert cleaned["packages"] == ""


def test_map_import_item_basic_fields():
    item = {
        "item_ser_no": 1, "product_description": "VALVE", "part_number": "",
        "model_or_type": "X200", "order_number": "", "hsn_code": "84819000",
        "quantity": 5, "unit_of_measurement": "NOS", "unit_price": 20.0,
        "line_total": 100.0, "end_use_code": "", "fta_code": "",
        "country_of_origin": "DE", "igst_amount": "", "igst_rate": "", "packages": "",
    }
    mapped = _map_import_item(item, {})
    assert mapped["Item_Desc1"] == "VALVE"
    assert mapped["Item_Country_Org"] == "DE"
    assert mapped["Model"] == "X200"
    # line_total (already extracted for every import item, e.g. from a
    # source "Extended Price" column) now reaches the BOE sheet instead
    # of being silently discarded -- see Item_Taxable_Val on the template.
    assert mapped["Item_Taxable_Val"] == 100.0


def test_clean_invoice_header_export_still_repairs_gstin():
    warnings = []
    clean = _clean_invoice_header(
        {"supplier_tax_id": "03AABCS7623P1Z3X"}, warnings, "export",
    )
    # Trailing stray char trimmed back to the valid 15-char shape, as before.
    assert clean["supplier_tax_id"] == "03AABCS7623P1Z3"
    assert not warnings


def test_clean_invoice_header_import_does_not_gstin_check_foreign_vat():
    # A European supplier's VAT number on an import invoice isn't a GSTIN
    # and has no fixed shape to validate against -- it must pass through
    # untouched, with no "doesn't match GSTIN shape" warning.
    warnings = []
    clean = _clean_invoice_header(
        {"supplier_tax_id": "DE123456789"}, warnings, "import",
    )
    assert clean["supplier_tax_id"] == "DE123456789"
    assert not warnings


def test_clean_invoice_header_import_state_code_not_overridden_by_foreign_vat():
    # state_code cross-check reuses supplier_tax_id's first two digits, which
    # only makes sense when supplier_tax_id really is an Indian GSTIN
    # (export). A numeric-leading foreign VAT number ("12..." here, e.g. a
    # non-hyphenated EU VAT) must not get misread as a "27" state prefix
    # and stomp an already-correct state_code on an import invoice.
    warnings = []
    clean = _clean_invoice_header(
        {"supplier_tax_id": "12345678900", "state_code": "27"}, warnings, "import",
    )
    assert clean["state_code"] == "27"
    assert not warnings


def test_validate_and_coerce_export_no_confidence_and_blank_missing():
    schema = load_template_schema(_TEMPLATES / "EXP_TEMPLET.xlsx")
    raw = {
        "invoice_header": {"invoice_number": "INV-1", "supplier_name": "ACME"},
        "line_items": [
            {"item_ser_no": 1, "product_description": "WIDGET", "quantity": 2, "unit_price": 10.0, "line_total": 20.0},
        ],
    }
    sheets, warnings, header = validate_and_coerce(raw, schema, "export")

    assert "ITEM" in sheets and len(sheets["ITEM"]) == 1
    row = sheets["ITEM"][0]

    # Every row carries every template column ...
    for field_name in schema["ITEM"]["fields"]:
        assert field_name in row
    # ... and anything not extracted is "" -- not "N", not "P", not dropped.
    assert row["item_Cess"] == ""
    assert row["Printed"] == ""
    # No confidence tracking anywhere.
    assert "_confidence" not in row
    assert not any(k for k in header if "confidence" in k.lower())


# ── line_total is never invented from qty x price ──────────────────────

def test_missing_line_total_is_left_blank_not_computed():
    warnings = []
    item = _clean_generic_item(
        {"product_description": "WIDGET", "quantity": 5, "unit_price": 10.0}, warnings, 0, 1,
    )
    assert item["line_total"] == ""


def test_free_of_cost_zero_line_total_is_kept_as_zero_with_warning():
    warnings = []
    item = _clean_generic_item(
        {"product_description": "SAMPLE", "quantity": 5, "unit_price": 10.0, "line_total": 0},
        warnings, 0, 1,
    )
    assert item["line_total"] == 0
    assert any("free of cost" in w for w in warnings)


def test_zero_line_total_survives_mapping_into_the_export_row():
    # The 0 must reach the final template row as 0, not get blanked to ""
    # (or recomputed) anywhere between cleaning and model validation.
    schema = load_template_schema(_TEMPLATES / "EXP_TEMPLET.xlsx")
    raw = {
        "invoice_header": {"invoice_number": "INV-1"},
        "line_items": [
            {"item_ser_no": 1, "product_description": "SAMPLE", "quantity": 5, "unit_price": 10.0, "line_total": 0},
            {"item_ser_no": 2, "product_description": "WIDGET", "quantity": 5, "unit_price": 10.0},
        ],
    }
    sheets, _, _ = validate_and_coerce(raw, schema, "export")
    assert [row["ItmTaxableVal"] for row in sheets["ITEM"]] == [0, ""]


def test_validate_and_coerce_import_uses_boe_sheet():
    schema = load_template_schema(_TEMPLATES / "IMP_TEMPLET.xlsx")
    raw = {
        "invoice_header": {"invoice_number": "INV-2"},
        "line_items": [{"item_ser_no": 1, "product_description": "PUMP", "quantity": 1, "unit_price": 100.0}],
    }
    sheets, warnings, header = validate_and_coerce(raw, schema, "import")
    assert "BOE" in sheets and len(sheets["BOE"]) == 1
    assert "ITEM" not in sheets


# ── item_ser_no reset backstop: confirmed real case, 2607510.pdf1.pdf ──
# One page's items (out of many pages that correctly came back null,
# letting the clean 1..N counter fill in) still got a hallucinated LOCAL
# count restarting at 1, landing mid-list as ...,79,80,1,2,...,22,92,...
# instead of a continuing sequence.

def test_renumber_items_if_ser_no_resets_detects_and_fixes_mid_list_restart():
    items = [
        {"item_ser_no": 79}, {"item_ser_no": 80},
        {"item_ser_no": 1}, {"item_ser_no": 2}, {"item_ser_no": 3},
        {"item_ser_no": 92},
    ]
    warnings = []
    _renumber_items_if_ser_no_resets(items, warnings)
    assert [it["item_ser_no"] for it in items] == [1, 2, 3, 4, 5, 6]
    assert warnings and "reset or repeated" in warnings[0]


def test_renumber_items_if_ser_no_resets_leaves_clean_ascending_sequence_alone():
    items = [{"item_ser_no": 1}, {"item_ser_no": 2}, {"item_ser_no": None}, {"item_ser_no": 4}]
    warnings = []
    _renumber_items_if_ser_no_resets(items, warnings)
    assert [it["item_ser_no"] for it in items] == [1, 2, None, 4]
    assert not warnings


def test_renumber_items_if_ser_no_resets_leaves_all_null_alone():
    # The common case: no real SNo column at all, so item_ser_no is null
    # on every item and validate_and_coerce()'s own counter already fills
    # in a clean sequence upstream -- nothing here to detect or touch.
    items = [{"item_ser_no": None}, {"item_ser_no": None}, {"item_ser_no": None}]
    warnings = []
    _renumber_items_if_ser_no_resets(items, warnings)
    assert [it["item_ser_no"] for it in items] == [None, None, None]
    assert not warnings


def test_validate_and_coerce_renumbers_end_to_end_on_ser_no_reset():
    schema = load_template_schema(_TEMPLATES / "IMP_TEMPLET.xlsx")
    raw = {
        "invoice_header": {"invoice_number": "INV-3"},
        "line_items": [
            {"item_ser_no": None, "product_description": "A", "quantity": 1, "unit_price": 1.0},
            {"item_ser_no": None, "product_description": "B", "quantity": 1, "unit_price": 1.0},
            {"item_ser_no": 1, "product_description": "C", "quantity": 1, "unit_price": 1.0},
            {"item_ser_no": 2, "product_description": "D", "quantity": 1, "unit_price": 1.0},
        ],
    }
    sheets, warnings, header = validate_and_coerce(raw, schema, "import")
    assert [row["SL_No"] for row in sheets["BOE"]] == [1, 2, 3, 4]
    assert any("reset or repeated" in w for w in warnings)


# ── state_code is only cross-checked against a REAL (shape-valid) GSTIN ──

def test_state_code_not_overridden_by_a_malformed_tax_id_starting_with_digits():
    # An IEC (or any mis-read) in supplier_tax_id that merely STARTS with
    # two digits is not a GSTIN and must not stomp a correct state_code.
    warnings = []
    clean = _clean_invoice_header(
        {"supplier_tax_id": "0388012345", "state_code": "MH(27)"}, warnings, "export",
    )
    assert clean["state_code"] == "27"
    assert not any("used the GSTIN" in w for w in warnings)


def test_state_code_still_overridden_by_a_valid_gstin_on_mismatch():
    warnings = []
    clean = _clean_invoice_header(
        {"supplier_tax_id": "27AAACG8030H2ZR", "state_code": "29"}, warnings, "export",
    )
    assert clean["state_code"] == "27"
    assert any("used the GSTIN" in w for w in warnings)


def test_state_code_filled_from_a_valid_gstin_when_missing():
    clean = _clean_invoice_header({"supplier_tax_id": "27AAACG8030H2ZR"}, [], "export")
    assert clean["state_code"] == "27"


def test_gstin_with_a_letter_entity_code_is_valid():
    # 13th character is the registration number within the state: a LETTER
    # after the first nine. A digits-only pattern rejected these.
    from validate import _GSTIN_RE
    assert _GSTIN_RE.match("27AAACG8030HAZR")
    assert _GSTIN_RE.match("27AAACG8030H2ZR")
    assert not _GSTIN_RE.match("27AAACG8030H2XR")  # 14th char must be Z
    assert not _GSTIN_RE.match("0388012345")


# ── currency: printed names/symbols are normalized to ISO codes ─────────

def test_euro_is_normalized_to_eur():
    # Confirmed real case: "EURO" failed the validator's ISO check, which is
    # retryable, wasting a retry (and on one file the retry blanked it).
    assert _clean_invoice_header({"currency": "EURO"}, [], "export")["currency"] == "EUR"


def test_other_unambiguous_currency_spellings_are_normalized():
    for raw, iso in [("Euros", "EUR"), ("€", "EUR"), ("US$", "USD"), ("U.S. Dollars", "USD"),
                     ("Rs.", "INR"), ("Indian Rupees", "INR"), ("£", "GBP"), ("usd", "USD")]:
        assert _clean_invoice_header({"currency": raw}, [], "export")["currency"] == iso, raw


def test_ambiguous_or_unknown_currency_is_left_alone():
    # "$" and "¥" each belong to several currencies -- not guessed.
    for raw in ("$", "¥", "DOLLARS", "ZZ-COIN"):
        assert _clean_invoice_header({"currency": raw}, [], "export")["currency"] == raw, raw


def test_missing_currency_stays_blank():
    assert _clean_invoice_header({"currency": None}, [], "export")["currency"] == ""
    assert _clean_invoice_header({}, [], "export")["currency"] == ""


# ── packing-list repeats and bundle duplicates ──────────────────────────

from validate import _drop_packing_list_duplicates, _drop_duplicates_that_break_total


def _gi(desc, qty=475, price=65.0, total=30875.0, part=""):
    return {"product_description": desc, "quantity": qty, "unit_price": price,
            "line_total": total, "part_number": part, "item_ser_no": 1}


def test_priceless_packing_list_repeat_is_dropped_by_description_without_a_part_number():
    # Real case (20260829132622): no part-number column; the packing list
    # repeated the item with no price -> phantom row 2 failed validation.
    items = [_gi("OMEPRAZOLE GR PELLETS 13.4% M/M"), _gi("OMEPRAZOLE GR PELLETS 13.4% M/M", price="", total="")]
    warnings = []
    kept = _drop_packing_list_duplicates(items, warnings)
    assert len(kept) == 1 and kept[0]["unit_price"] == 65.0
    assert warnings and "packing-list repeat" in warnings[0]


def test_priceless_repeat_tolerates_small_ocr_differences_in_the_description():
    items = [_gi("ATORVASTATIN CALCIUM TRIHYDRATE USP", qty=26, price=120.0, total=3120.0),
             _gi("ATORVASTATINCALCIUM TRIHYDRATE USP", qty=26, price="", total="")]
    assert len(_drop_packing_list_duplicates(items, [])) == 1


def test_priceless_row_for_a_different_product_is_kept():
    items = [_gi("OMEPRAZOLE GR PELLETS"), _gi("PANTOPRAZOLE SODIUM SESQUIHYDRATE", price="", total="")]
    assert len(_drop_packing_list_duplicates(items, [])) == 2


def test_priceless_row_with_a_different_quantity_is_kept():
    items = [_gi("OMEPRAZOLE GR PELLETS"), _gi("OMEPRAZOLE GR PELLETS", qty=100, price="", total="")]
    assert len(_drop_packing_list_duplicates(items, [])) == 2


def _bundle_items():
    # Real case (57_202608281144371.pdf, an 18-page bundle): the same
    # 30 KG x 80.00 = 2,400.00 line from the commercial invoice, the
    # purchase order and the export invoice.
    return [
        _gi("AMLODIPINE BESYLATE PH.EUR", 30, 80.0, 2400.0),
        _gi("Amiodipine Besilate-Hetero", 30, 80.0, 2400.0),
        _gi("JAMLODIPIN R93339 BESILATE P1 PH.EUR Batch: AAAL 6", 30.0, 80.0, 2400.0),
    ]


def test_bundle_duplicates_are_dropped_when_that_reconciles_the_printed_total():
    warnings = []
    kept = _drop_duplicates_that_break_total(_bundle_items(), {"invoice_total": 2400}, warnings)
    assert len(kept) == 1 and kept[0]["product_description"] == "AMLODIPINE BESYLATE PH.EUR"
    assert warnings and "bundled PDF" in warnings[0]


def test_genuine_repeated_lines_are_kept_when_they_already_reconcile():
    # Two batches, same quantity and price: lines sum to the printed total,
    # so nothing is dropped no matter how similar they look.
    items = [_gi("OMEPRAZOLE GR PELLETS batch A"), _gi("OMEPRAZOLE GR PELLETS batch B")]
    assert len(_drop_duplicates_that_break_total(items, {"invoice_total": 61750}, [])) == 2


def test_duplicates_are_kept_if_dropping_them_would_not_reconcile_either():
    # Total matches neither 3 x 2400 nor any smaller subset -> a human
    # needs to look; don't guess.
    assert len(_drop_duplicates_that_break_total(_bundle_items(), {"invoice_total": 3000}, [])) == 3


def test_no_numeric_total_means_nothing_is_dropped():
    assert len(_drop_duplicates_that_break_total(_bundle_items(), {"invoice_total": ""}, [])) == 3
    assert len(_drop_duplicates_that_break_total(_bundle_items(), {"invoice_total": "2400.00 / 2400.00"}, [])) == 3


def test_identical_numbers_on_unrelated_products_are_never_treated_as_duplicates():
    items = [_gi("OMEPRAZOLE GR PELLETS", 30, 80.0, 2400.0), _gi("ZZZZ QQQQ XXXX", 30, 80.0, 2400.0)]
    assert len(_drop_duplicates_that_break_total(items, {"invoice_total": 2400}, [])) == 2


def test_validate_and_coerce_applies_the_bundle_dedupe_end_to_end():
    schema = load_template_schema(_TEMPLATES / "EXP_TEMPLET.xlsx")
    raw = {
        "invoice_header": {"invoice_number": "SI3626102057", "invoice_total": 2400},
        "line_items": [
            {"item_ser_no": 1, "product_description": "AMLODIPINE BESYLATE PH.EUR", "quantity": 30, "unit_price": 80.0, "line_total": 2400.0},
            {"item_ser_no": 2, "product_description": "Amiodipine Besilate-Hetero", "quantity": 30, "unit_price": 80.0, "line_total": 2400.0},
            {"item_ser_no": 3, "product_description": "AMLODIPINE BESILATE PH EUR", "quantity": 30, "unit_price": 80.0, "line_total": 2400.0},
        ],
    }
    sheets, warnings, _ = validate_and_coerce(raw, schema, "export")
    assert len(sheets["ITEM"]) == 1
    assert [row["Item_Ser_No"] for row in sheets["ITEM"]] == [1]


def test_bundle_duplicate_with_a_long_description_is_still_recognised():
    # The REAL third description of the 18-page bundle: the same item with
    # batch / dates / packing detail appended. Whole-string similarity to
    # the short one is only ~0.3; word overlap is what recognises it.
    items = _bundle_items()
    items[2]["product_description"] = (
        "JAMLODIPIN R93339 BESILATE P1 PH.EUR Batch: AAAL 6070071 Mfg.Date: 62026 "
        "Expiry/Retest Date: 62031 Packing Details: 1X30.00 KGS"
    )
    kept = _drop_duplicates_that_break_total(items, {"invoice_total": 2400}, [])
    assert len(kept) == 1


def test_word_overlap_does_not_match_unrelated_products():
    from validate import _desc_similarity
    assert _desc_similarity("OMEPRAZOLE GR PELLETS 13.4% M/M", "PANTOPRAZOLE SODIUM SESQUIHYDRATE") < 0.6
    assert _desc_similarity("BOLT", "NUT WASHER ASSEMBLY KIT") < 0.6


def test_bundle_duplicates_with_different_extra_text_are_recognised_by_shared_words():
    # The REAL second run of the 18-page bundle: each document appends
    # different text to the same drug name, so fractional overlap is low but
    # the two long drug-name words are shared.
    items = [
        _gi("AMLODIPINE BESYLATE PH.EUR [1X30 KGS]", 30.0, 80.0, 2400.0),
        _gi("Amiodipine Besilate-Hetero Mfg Part No:11101355-HETERO", 30, 80.0, 2400.0),
        _gi("JAMLODIPIN R93339 BESILATE", 30.0, 80.0, 2400.0),
    ]
    warnings = []
    kept = _drop_duplicates_that_break_total(items, {"invoice_total": 2400}, warnings)
    assert len(kept) == 1 and kept[0]["product_description"].startswith("AMLODIPINE")
    assert warnings and "rows 2, 3" in warnings[0]


def test_shared_words_alone_never_drop_anything_that_already_reconciles():
    items = [_gi("AMLODIPINE BESYLATE batch A", 30, 80.0, 2400.0), _gi("AMLODIPINE BESYLATE batch B", 30, 80.0, 2400.0)]
    assert len(_drop_duplicates_that_break_total(items, {"invoice_total": 4800}, [])) == 2


def test_shared_long_words_ignores_short_and_unrelated_words():
    from validate import _shared_long_words
    assert _shared_long_words("PH EUR KGS", "PH EUR KGS") == 0           # all words under 5 chars
    assert _shared_long_words("OMEPRAZOLE GR PELLETS", "PANTOPRAZOLE SODIUM SESQUIHYDRATE") == 0
    assert _shared_long_words("AMLODIPINE BESYLATE PH.EUR", "Amiodipine Besilate-Hetero") == 2


# ── a row with no quantity, price or amount is a stray fragment ─────────

def test_a_row_with_no_numbers_at_all_is_discarded_with_a_warning():
    warnings = []
    assert _clean_generic_item({"product_description": "STRAY FRAGMENT", "hsn_code": ""}, warnings, 4, 5) is None
    assert warnings and "discarded a row with no quantity, unit price or amount" in warnings[0]


def test_a_row_with_any_one_number_is_kept():
    # Even a lone quantity is a real (if incomplete) row: let the validator
    # report what's missing instead of silently dropping it.
    for field, value in [("quantity", 5), ("unit_price", 2.5), ("line_total", 12.5)]:
        item = _clean_generic_item({"product_description": "WIDGET", field: value}, [], 0, 1)
        assert item is not None, field
