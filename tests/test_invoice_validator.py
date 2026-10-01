"""
Tests for invoice_validator.py's deterministic business-rule checks.
Uses the two extraction-result shapes from the spec (export ITEM sheet,
import BOE sheet) as fixtures, plus small targeted variations per rule.
"""
from __future__ import annotations

from decimal import Decimal

from invoice_validator import (
    validate_invoice,
    validate_line_items,
    validate_totals,
    validate_taxes,
    validate_required_fields,
    validate_cross_fields,
    validate_source_consistency,
    validate_format_and_types,
    build_retry_feedback,
    _parse_header_total,
    RETRYABLE_RULES,
)


def _export_result(item_overrides: dict | None = None, header_overrides: dict | None = None) -> dict:
    item = {
        "Item_Ser_No": 1,
        "Item_Desc": "BOTTLED CHEMICAL",
        "Item_Qty": 5184,
        "Item_Unit_Price": 9.87,
        "ItmTaxableVal": 51166.08,
        "Item_Unit1": "BOT",
        "Qty Tariff": 5184,
        "Unit Tariff": "BOT",
        "part_number": "1904945",
        "order_number": "4500907422",
        "model_or_type": "",
        "Itm_Source_Cntry": "IN",
        "State Code": "",
        "ItmTotpkg": "",
    }
    if item_overrides:
        item.update(item_overrides)
    header = {
        "invoice_total": 51166.08,
        "currency": "USD",
        "country_of_origin": "IN",
        "country_of_destination": "USA",
        "invoice_number": "INV-1",
        "invoice_date": "2026-08-08",
        "supplier_tax_id": "03AABCS7623P1Z3",
    }
    if header_overrides:
        header.update(header_overrides)
    return {"header": header, "sheets": {"ITEM": [item]}}


def _import_result(item_overrides: dict | None = None, header_overrides: dict | None = None) -> dict:
    item = {
        "SL_No": 1,
        "Item_Desc1": "ASSEMBLY UNIT",
        "Item_Qty": 1,
        "Item_Unit": "SET",
        "Item_Unit_Price": 1116596.07,
        "Qty_Tariff": 1,
        "Unit_Tariff": "SET",
        "Item_Country_Org": "GERMANY",
        "COO Country": "GERMANY",
        "part_number": "100-046567-001",
        "order_number": "IN66265187",
        "model_or_type": "",
    }
    if item_overrides:
        item.update(item_overrides)
    header = {
        "invoice_total": 1116596.07,
        "currency": "USD",
        "country_of_origin": "GERMANY",
        "invoice_number": "IMP-1",
        "invoice_date": "2026-08-08",
    }
    if header_overrides:
        header.update(header_overrides)
    return {"header": header, "sheets": {"BOE": [item]}}


def _find(checks, rule, item=None):
    for c in checks:
        if c.rule == rule and c.item == item:
            return c
    return None


# ── 1 & 2: Quantity x Unit Price ────────────────────────────────────────

def test_correct_qty_x_rate_passes():
    items = [{"Item_Qty": 5184, "Item_Unit_Price": 9.87, "ItmTaxableVal": 51166.08}]
    checks = validate_line_items(items, "export")
    c = _find(checks, "QTY_X_UNIT_PRICE", item=1)
    assert c.status == "PASS"


def test_incorrect_qty_x_rate_errors():
    items = [{"Item_Qty": 5184, "Item_Unit_Price": 9.87, "ItmTaxableVal": 51000}]
    checks = validate_line_items(items, "export")
    c = _find(checks, "QTY_X_UNIT_PRICE", item=1)
    assert c.status == "ERROR"
    assert c.retryable is True
    assert "Quantity" in c.message and "Unit Price" in c.message


# ── 3 & 4: Header total vs line totals ──────────────────────────────────

def test_correct_header_total_passes():
    header = {"invoice_total": 51166.08}
    items = [{"ItmTaxableVal": 51166.08}]
    checks = validate_totals(header, items, "export")
    c = _find(checks, "HEADER_LINE_TOTAL")
    assert c.status == "PASS"


def test_incorrect_header_total_fails_but_is_not_retryable():
    # ERROR (fails validation -- a human needs to look at it), but
    # deliberately NOT retryable: confirmed real case (Piramal export
    # invoice) where the header total genuinely did NOT match the sum of
    # its own line items because the SOURCE DOCUMENT's own printed total
    # only covered the first item -- a real document quirk, not an
    # extraction mistake. Forcing this retryable would let a retry "fix"
    # an already-correct 51166.08 into the wrong 78382.08 just to match
    # the sum, which is exactly what happened when this was retryable
    # before. A plain WARNING (tried next) swung too far the other way --
    # it let a confirmed hallucinated total (not derived from a sum
    # mismatch at all, see test_invoice_validator.py's
    # MULTI_INVOICE_TOTAL_SPLIT tests) reach "valid" output with no
    # signal. ERROR + not-in-RETRYABLE_RULES is the combination that
    # surfaces every mismatch for review without ever auto-"fixing" one.
    header = {"invoice_total": 51166.08}
    items = [{"ItmTaxableVal": 51166.08}, {"ItmTaxableVal": 27216.00}]
    checks = validate_totals(header, items, "export")
    c = _find(checks, "HEADER_LINE_TOTAL")
    assert c.status == "ERROR"
    assert c.retryable is False
    assert "HEADER_LINE_TOTAL" not in RETRYABLE_RULES
    assert c.expected == Decimal("78382.08")
    assert c.actual == Decimal("51166.08")
    assert c.difference == Decimal("27216.00")
    assert "78382.08" in c.message and "51166.08" in c.message and "27216.00" in c.message


# ── 5: Rounding difference within tolerance ─────────────────────────────

def test_rounding_difference_within_tolerance_passes():
    # 100 * 9.876 = 987.60; extracted 987.62 -- 0.02 off, under the 0.05 default tolerance
    items = [{"Item_Qty": 100, "Item_Unit_Price": 9.876, "ItmTaxableVal": 987.62}]
    checks = validate_line_items(items, "export")
    c = _find(checks, "QTY_X_UNIT_PRICE", item=1)
    assert c.status == "PASS"
    assert c.difference == Decimal("0.02")


def test_difference_beyond_tolerance_errors():
    items = [{"Item_Qty": 100, "Item_Unit_Price": 9.876, "ItmTaxableVal": 988.00}]
    checks = validate_line_items(items, "export")
    c = _find(checks, "QTY_X_UNIT_PRICE", item=1)
    assert c.status == "ERROR"


# ── Item_Taxable_Val on import: added to BOE (templates/IMP_TEMPLET.xlsx)
# and wired into _ITEM_FIELDS so QTY_X_UNIT_PRICE / HEADER_LINE_TOTAL can
# actually run for import invoices -- confirmed real case: 2607510.pdf1.pdf
# had a real per-line "Extended Price" the model was already extracting
# into line_total, but it was silently discarded because BOE had no column
# to map it into, so both rules were unconditionally NOT_CHECKED for every
# import invoice this pipeline processed.

def test_import_correct_qty_x_rate_passes():
    items = [{"Item_Qty": 10, "Item_Unit_Price": 1.0199, "Item_Taxable_Val": 10.20}]
    checks = validate_line_items(items, "import")
    c = _find(checks, "QTY_X_UNIT_PRICE", item=1)
    assert c.status == "PASS"


def test_import_incorrect_qty_x_rate_errors():
    items = [{"Item_Qty": 10, "Item_Unit_Price": 1.0199, "Item_Taxable_Val": 5.00}]
    checks = validate_line_items(items, "import")
    c = _find(checks, "QTY_X_UNIT_PRICE", item=1)
    assert c.status == "ERROR"
    assert c.retryable is True


def test_import_correct_header_total_passes():
    header = {"invoice_total": 18.36}
    items = [{"Item_Taxable_Val": 10.20}, {"Item_Taxable_Val": 8.16}]
    checks = validate_totals(header, items, "import")
    c = _find(checks, "HEADER_LINE_TOTAL")
    assert c.status == "PASS"


def test_import_tax_reconciliation_stays_not_checked():
    # BOE has Item_Taxable_Val now, but still no per-item IGST rate/amount
    # columns (only notification-number reference fields) -- this
    # three-way check must stay NOT_CHECKED for import regardless.
    items = [{"Item_Taxable_Val": 10.20}]
    checks = validate_taxes(items, "import")
    c = _find(checks, "TAX_RECONCILIATION")
    assert c.status == "NOT_CHECKED"


# ── 6: Missing optional field is NOT an automatic error ─────────────────

def test_missing_optional_model_with_no_source_label_is_not_checked():
    items = [{"Item_Desc": "PLAIN ITEM WITH NO MODEL LABEL", "model_or_type": "", "part_number": "", "order_number": ""}]
    checks = validate_source_consistency(items, "export", invoice_text="")
    c = _find(checks, "MODEL_SOURCE_MATCH", item=1)
    assert c.status == "NOT_CHECKED"


def test_model_named_in_description_but_field_empty_is_warning_not_error():
    items = [{"Item_Desc": "PUMP UNIT Model Number: HP78h/HP89j", "model_or_type": "", "part_number": "", "order_number": ""}]
    checks = validate_source_consistency(items, "export", invoice_text="")
    c = _find(checks, "MODEL_SOURCE_MATCH", item=1)
    assert c.status == "WARNING"
    assert c.retryable is False


# ── 7: Missing required field ───────────────────────────────────────────

def test_missing_required_field_errors():
    items = [{"Item_Ser_No": 1, "Item_Desc": "X", "Item_Qty": "", "Item_Unit1": "PCS", "Item_Unit_Price": 5}]
    checks = validate_required_fields(items, "export")
    c = _find(checks, "REQUIRED_FIELD_MISSING", item=1)
    # multiple required-field checks exist per item; find the quantity one specifically
    qty_check = next(c for c in checks if c.item == 1 and c.field == "Item_Qty")
    assert qty_check.status == "ERROR"
    assert qty_check.retryable is True


def test_all_required_fields_present_passes():
    items = [{"Item_Ser_No": 1, "Item_Desc": "X", "Item_Qty": 5, "Item_Unit1": "PCS", "Item_Unit_Price": 5}]
    checks = validate_required_fields(items, "export")
    assert all(c.status == "PASS" for c in checks if c.item == 1)


def test_missing_unit_is_not_flagged_or_retried():
    # Confirmed real case: a 10-page export invoice had no unit-of-measure
    # column printed anywhere in the document -- quantity/unit_price/line
    # amount were all extracted correctly, but "unit" genuinely doesn't
    # exist on the source. Unlike quantity/unit_price/description/ser_no,
    # a missing unit must never produce a REQUIRED_FIELD_MISSING check at
    # all (no ERROR, no PASS) -- retrying it would burn attempts that can
    # never succeed.
    items = [{"Item_Ser_No": 1, "Item_Desc": "X", "Item_Qty": 5, "Item_Unit1": "", "Item_Unit_Price": 5}]
    checks = validate_required_fields(items, "export")
    assert not any(c.field == "Item_Unit1" for c in checks)


# ── 8 & 9: Part number vs source text ───────────────────────────────────

def test_part_number_matches_source_passes():
    items = [{"Item_Desc1": "ITEM P/N: 100-046567-001", "part_number": "100-046567-001", "model_or_type": "", "order_number": ""}]
    checks = validate_source_consistency(items, "import", invoice_text="")
    c = _find(checks, "PART_NUMBER_SOURCE_MATCH", item=1)
    assert c.status == "PASS"


def test_part_number_mismatch_errors():
    items = [{"Item_Desc1": "ITEM P/N: 100-046567-001", "part_number": "100-046567-002", "model_or_type": "", "order_number": ""}]
    checks = validate_source_consistency(items, "import", invoice_text="")
    c = _find(checks, "PART_NUMBER_SOURCE_MATCH", item=1)
    assert c.status == "ERROR"
    assert c.retryable is True
    assert c.expected == "100-046567-001"
    assert c.actual == "100-046567-002"


# ── 10 & 11: BOE (import) extraction ────────────────────────────────────

def test_correct_boe_extraction_has_no_errors():
    result = _import_result()
    validation = validate_invoice(result, invoice_text="P/O: IN66265187", shipment_type="import")
    assert validation.valid is True
    assert validation.errors == []
    # BOE has no per-line amount column -- confirm this is why, not silence
    qty_price_checks = [c for c in validation.checks if c.rule == "QTY_X_UNIT_PRICE"]
    assert all(c.status == "NOT_CHECKED" for c in qty_price_checks)


def test_incorrect_boe_quantity_vs_tariff_quantity_errors():
    # BOE has no line-amount column, so the qty/rate mismatch this rule
    # targets shows up as a quantity vs tariff-quantity mismatch instead
    # -- both are populated, independently, on BOE.
    result = _import_result(item_overrides={"Item_Qty": 5, "Qty_Tariff": 1})
    validation = validate_invoice(result, invoice_text="", shipment_type="import")
    assert validation.valid is False
    c = _find(validation.checks, "QTY_VS_TARIFF_QTY", item=1)
    assert c.status == "ERROR"


# ── 12: Country mismatch ────────────────────────────────────────────────

def test_country_of_origin_mismatch_is_warning():
    header = {"country_of_origin": "IN"}
    items = [{"Itm_Source_Cntry": "GERMANY"}]
    checks = validate_cross_fields(header, items, "export")
    c = _find(checks, "COUNTRY_OF_ORIGIN_CONSISTENCY", item=1)
    assert c.status == "WARNING"


def test_country_of_origin_match_passes():
    header = {"country_of_origin": "IN"}
    items = [{"Itm_Source_Cntry": "IN"}]
    checks = validate_cross_fields(header, items, "export")
    c = _find(checks, "COUNTRY_OF_ORIGIN_CONSISTENCY", item=1)
    assert c.status == "PASS"


# ── 13: Unit mismatch ───────────────────────────────────────────────────

def test_unit_vs_tariff_unit_mismatch_is_warning():
    items = [{"Item_Unit1": "PCS", "Qty Tariff": 1, "Unit Tariff": "KG", "Item_Qty": 1}]
    checks = validate_cross_fields({}, items, "export")
    c = _find(checks, "UNIT_VS_TARIFF_UNIT", item=1)
    assert c.status == "WARNING"


# ── 14: Package count -- sum for genuine per-item counts, equality for  ──
# ── header-copy fallbacks, never a blind sum of repeated fallbacks ─────

def test_package_counts_reconcile_by_sum_when_items_have_their_own_counts():
    # Confirmed real case: two items, 108 and 504 packages on different
    # pallet ranges, header.total_packages 612 (108 + 504) -- each item
    # keeps its own distinct number, and the check reconciles by SUM.
    header = {"total_packages": 612}
    items = [{"ItmTotpkg": 108}, {"ItmTotpkg": 504}]
    checks = validate_cross_fields(header, items, "export")
    c = _find(checks, "PACKAGE_COUNT_CONSISTENCY")
    assert c.status == "PASS"


def test_package_counts_pass_when_every_row_is_the_same_header_copy_fallback():
    # No per-item breakdown printed anywhere -- every row is validate.py's
    # header-copy fallback. Summing these (12) would wrongly overcount the
    # true total (6) by a factor of 2; per-row equality against the header
    # is the right check here instead.
    header = {"total_packages": 6}
    items = [{"ItmTotpkg": 6}, {"ItmTotpkg": 6}]
    checks = validate_cross_fields(header, items, "export")
    c = _find(checks, "PACKAGE_COUNT_CONSISTENCY")
    assert c.status == "PASS"


def test_package_counts_warn_on_genuine_mismatch():
    # Neither the sum (300) nor per-row equality against the header (250)
    # holds -- a genuine inconsistency worth flagging.
    header = {"total_packages": 250}
    items = [{"ItmTotpkg": 100}, {"ItmTotpkg": 200}]
    checks = validate_cross_fields(header, items, "export")
    c = _find(checks, "PACKAGE_COUNT_CONSISTENCY")
    assert c.status == "WARNING"
    assert c.actual == Decimal("300")


def test_package_counts_not_checked_when_no_item_has_one():
    header = {"total_packages": 612}
    items = [{"ItmTotpkg": ""}, {"ItmTotpkg": ""}]
    checks = validate_cross_fields(header, items, "export")
    c = _find(checks, "PACKAGE_COUNT_CONSISTENCY")
    assert c.status == "NOT_CHECKED"


# ── Full-invoice orchestration: header-total fails, but never auto-retries ──

def test_validate_invoice_flags_header_total_mismatch_as_invalid_but_not_retryable_end_to_end():
    result = _export_result()
    result["sheets"]["ITEM"].append({
        **result["sheets"]["ITEM"][0],
        "Item_Ser_No": 2,
        "ItmTaxableVal": 27216.00,
        "Item_Qty": 100,
        "Item_Unit_Price": 272.16,  # 100 * 272.16 = 27216.00 exactly -- QTY_X_UNIT_PRICE PASSes, isolating the header-level check
        "Qty Tariff": 100,  # keep QTY_VS_TARIFF_QTY consistent with the new Item_Qty
    })
    # header.invoice_total (51166.08) now understates the true sum (78382.08)
    # -- ERROR, so the document must fail validation (not silently "valid"),
    # but must never show up as a retryable error (no auto re-extraction).
    validation = validate_invoice(result, invoice_text="", shipment_type="export")
    assert validation.valid is False
    c = _find(validation.checks, "HEADER_LINE_TOTAL")
    assert c.status == "ERROR"
    assert c.difference == Decimal("27216.00")
    assert c not in validation.retryable_errors


def test_not_checked_returned_when_inputs_missing_not_error():
    header = {"invoice_total": None}
    items = [{"ItmTaxableVal": None}]
    checks = validate_totals(header, items, "export")
    c = _find(checks, "HEADER_LINE_TOTAL")
    assert c.status == "NOT_CHECKED"


# ── MULTI_INVOICE_TOTAL_SPLIT: combined-document total must mirror ──────
# invoice_number's own split, never be a single computed/summed figure.

def test_single_invoice_total_not_split_when_combined_document_hallucinates_sum():
    # Confirmed real case: invoice_number correctly split into two
    # sub-invoices, but invoice_total came back as "5888" -- the SUM of
    # the two sub-invoices' real totals (4416.00 + 1472.00), not a value
    # printed anywhere on the document. No line-item amounts were
    # extracted at all, so HEADER_LINE_TOTAL alone (NOT_CHECKED when there's
    # nothing to sum) would have let this through completely unflagged.
    header = {"invoice_number": "944624267 / 944624264", "invoice_total": "5888"}
    items = [{"ItmTaxableVal": None}, {"ItmTaxableVal": None}]
    checks = validate_totals(header, items, "export")
    c = _find(checks, "MULTI_INVOICE_TOTAL_SPLIT")
    assert c.status == "ERROR"
    assert c.expected == "2 '/'-separated total(s), one per sub-invoice"
    assert c.actual == "5888"


def test_correctly_split_multi_invoice_total_passes():
    header = {"invoice_number": "IN2604006721 / IN2604006722", "invoice_total": "10721.49 / 29812.90"}
    items = [{"ItmTaxableVal": None}]
    checks = validate_totals(header, items, "export")
    assert _find(checks, "MULTI_INVOICE_TOTAL_SPLIT") is None


def test_single_invoice_number_with_single_total_is_unaffected():
    header = {"invoice_number": "944624267", "invoice_total": 5888}
    items = [{"ItmTaxableVal": None}]
    checks = validate_totals(header, items, "export")
    assert _find(checks, "MULTI_INVOICE_TOTAL_SPLIT") is None


def test_slashes_inside_a_single_invoice_number_are_not_a_combined_document():
    # Real invoice numbers from this project's own outputs -- a bare "/"
    # is part of the number itself, not a sub-invoice separator.
    for number in ("KA/2627/I/033552", "EXPU2/2627/0612"):
        header = {"invoice_number": number, "invoice_total": 4416.0}
        checks = validate_totals(header, [{"ItmTaxableVal": None}], "export")
        assert _find(checks, "MULTI_INVOICE_TOTAL_SPLIT") is None, number


def test_combined_document_of_slashed_invoice_numbers_counts_sub_invoices_correctly():
    # Two sub-invoices, each with slashes inside its own number: that's 2
    # parts, not 8.
    header = {
        "invoice_number": "KA/2627/I/000431 / KA/2627/I/000433",
        "invoice_total": "4416.00 / 1472.00",
    }
    checks = validate_totals(header, [{"ItmTaxableVal": None}], "export")
    assert _find(checks, "MULTI_INVOICE_TOTAL_SPLIT") is None
    assert _parse_header_total(header) == Decimal("5888.00")


def test_combined_document_of_slashed_invoice_numbers_still_flags_single_total():
    header = {"invoice_number": "KA/2627/I/000431 / KA/2627/I/000433", "invoice_total": "5888"}
    checks = validate_totals(header, [{"ItmTaxableVal": None}], "export")
    assert _find(checks, "MULTI_INVOICE_TOTAL_SPLIT").status == "ERROR"


def test_slashed_single_invoice_with_slashed_total_is_not_summed():
    header = {"invoice_number": "KA/2627/I/033552", "invoice_total": "100/200"}
    assert _parse_header_total(header) is None


def test_multi_invoice_total_split_mismatch_fails_validation_end_to_end():
    result = _export_result(header_overrides={
        "invoice_number": "944624267 / 944624264", "invoice_total": "5888",
    })
    result["sheets"]["ITEM"][0]["ItmTaxableVal"] = None
    validation = validate_invoice(result, invoice_text="", shipment_type="export")
    assert validation.valid is False
    c = _find(validation.checks, "MULTI_INVOICE_TOTAL_SPLIT")
    assert c.status == "ERROR"
    assert c in validation.retryable_errors


# ── Retry-feedback construction ─────────────────────────────────────────

def test_build_retry_feedback_routes_header_and_item_errors_separately():
    result = _export_result()
    result["sheets"]["ITEM"][0]["ItmTaxableVal"] = 999999  # forces QTY_X_UNIT_PRICE ERROR (item-level)
    result["header"]["invoice_number"] = ""  # forces REQUIRED_FIELD_MISSING ERROR (header-level)
    validation = validate_invoice(result, invoice_text="", shipment_type="export")
    assert validation.valid is False

    header_fb, items_fb = build_retry_feedback(validation)
    assert header_fb is not None and "REQUIRED_FIELD_MISSING" in header_fb
    assert items_fb is not None and "QTY_X_UNIT_PRICE" in items_fb
    assert "Do not invent missing information" in header_fb


def test_build_retry_feedback_returns_none_when_no_retryable_errors():
    result = _export_result()
    validation = validate_invoice(result, invoice_text="", shipment_type="export")
    assert validation.valid is True
    header_fb, items_fb = build_retry_feedback(validation)
    assert header_fb is None and items_fb is None


# ── Combined multi-sub-invoice documents (invoice_total as "A / B") ────

def test_combined_invoice_total_is_summed_when_invoice_number_matches():
    header = {"invoice_number": "IN2604006721 / IN2604006722", "invoice_total": "10721.49 / 29812.90"}
    assert _parse_header_total(header) == Decimal("40534.39")


def test_stray_slash_in_total_not_summed_on_a_single_invoice():
    # No "/" in invoice_number -- nothing indicates this is really a
    # combined document, so a stray slash in invoice_total should NOT be
    # silently summed (that would mask a genuine extraction error).
    header = {"invoice_number": "INV-1", "invoice_total": "10721.49 / 29812.90"}
    assert _parse_header_total(header) is None


def test_combined_invoice_total_reconciles_against_summed_line_items():
    header = {"invoice_number": "A / B", "invoice_total": "51166.08 / 27216.00"}
    items = [{"ItmTaxableVal": 51166.08}, {"ItmTaxableVal": 27216.00}]
    checks = validate_totals(header, items, "export")
    c = next(c for c in checks if c.rule == "HEADER_LINE_TOTAL")
    assert c.status == "PASS"


# ── supplier_tax_id shape: only real GSTIN on export, foreign tax ID on import ─

def test_export_supplier_tax_id_wrong_shape_warns():
    header = {"supplier_tax_id": "NOT-A-GSTIN"}
    checks = validate_format_and_types(header, [], "export")
    c = next(c for c in checks if c.rule == "FORMAT_TYPE" and c.field == "supplier_tax_id")
    assert c.status == "WARNING"


def test_import_foreign_vat_in_supplier_tax_id_field_is_not_checked():
    # On import, supplier_tax_id holds the foreign supplier's own tax ID
    # (VAT/EIN/...), which has no fixed shape -- it must not be flagged as
    # a malformed GSTIN just because it isn't a 15-char Indian GSTIN.
    header = {"supplier_tax_id": "DE123456789"}
    checks = validate_format_and_types(header, [], "import")
    c = next(c for c in checks if c.rule == "FORMAT_TYPE" and c.field == "supplier_tax_id")
    assert c.status == "NOT_CHECKED"
