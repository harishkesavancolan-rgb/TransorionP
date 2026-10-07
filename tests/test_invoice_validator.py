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


# ── Free-of-cost lines and missing line amounts ─────────────────────────

def test_zero_line_amount_with_real_price_is_a_non_retryable_warning():
    items = [{"Item_Qty": 5, "Item_Unit_Price": 10.0, "ItmTaxableVal": 0}]
    c = _find(validate_line_items(items, "export"), "QTY_X_UNIT_PRICE", item=1)
    assert c.status == "WARNING"
    assert c.retryable is False
    assert "free-of-cost" in c.message


def test_free_of_cost_line_does_not_fail_validation_end_to_end():
    result = _export_result()
    foc = dict(result["sheets"]["ITEM"][0], Item_Ser_No=2, ItmTaxableVal=0)
    result["sheets"]["ITEM"].append(foc)
    validation = validate_invoice(result, invoice_text="", shipment_type="export")
    assert not any(c.rule == "QTY_X_UNIT_PRICE" and c.status == "ERROR" for c in validation.checks)


def test_header_total_not_checked_against_a_partial_line_sum():
    # One of two lines has no amount: summing only the other would report
    # a guaranteed, misleading mismatch.
    header = {"invoice_total": 78382.08}
    items = [{"ItmTaxableVal": 51166.08}, {"ItmTaxableVal": None}]
    c = _find(validate_totals(header, items, "export"), "HEADER_LINE_TOTAL")
    assert c.status == "NOT_CHECKED"
    assert "1 of 2" in c.message


def test_header_total_includes_zero_value_lines_in_the_sum():
    header = {"invoice_total": 51166.08}
    items = [{"ItmTaxableVal": 51166.08}, {"ItmTaxableVal": 0}]
    c = _find(validate_totals(header, items, "export"), "HEADER_LINE_TOTAL")
    assert c.status == "PASS"


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


# ── ORDER_NUMBER_SOURCE_MATCH with several POs in one document ──────────

def test_each_items_po_is_matched_against_every_po_in_the_document():
    # Two POs printed on the invoice, one per item. The old check compared
    # both items to only the FIRST "Order No:" found and flagged item 2.
    text = "Order No: PO-111 ... later section ... Order No: PO-222"
    items = [
        {"Item_Desc": "A", "order_number": "PO-111"},
        {"Item_Desc": "B", "order_number": "PO-222"},
    ]
    checks = [c for c in validate_source_consistency(items, "export", text)
              if c.rule == "ORDER_NUMBER_SOURCE_MATCH"]
    assert [(c.item, c.status) for c in checks] == [(1, "PASS"), (2, "PASS")]


def test_a_po_that_matches_none_of_the_documents_pos_is_still_an_error():
    text = "Order No: PO-111 ... Order No: PO-222"
    items = [{"Item_Desc": "A", "order_number": "PO-999"}]
    c = next(c for c in validate_source_consistency(items, "export", text)
             if c.rule == "ORDER_NUMBER_SOURCE_MATCH")
    assert c.status == "ERROR" and c.retryable is True
    assert "PO-111" in c.message and "PO-222" in c.message


def test_order_number_with_dates_still_matches_by_containment():
    text = "P/O: 446297.1 and P/O: 450044.2"
    items = [{"Item_Desc": "A", "order_number": "446297.1 (20-APR-26) / 450044.2 (15-JUN-26)"}]
    c = next(c for c in validate_source_consistency(items, "export", text)
             if c.rule == "ORDER_NUMBER_SOURCE_MATCH")
    assert c.status == "PASS"


def test_order_check_is_not_checked_when_the_document_has_no_po_label():
    items = [{"Item_Desc": "A", "order_number": "PO-111"}]
    c = next(c for c in validate_source_consistency(items, "export", "no labels here")
             if c.rule == "ORDER_NUMBER_SOURCE_MATCH")
    assert c.status == "NOT_CHECKED"


# ── HEADER_LINE_TOTAL: discount / freight lines printed on the invoice ──
# Confirmed real case: 10 of 10 invoices from one exporter failed this
# check solely because they print "TOTAL / LESS TRADE DISCOUNT / ADD AIR
# FREIGHT / G-Total" -- every extracted total was correct.

# Layout text of the real 6168-CFWD invoice's totals block (abridged).
_ADJUSTED_TOTALS_TEXT = """\
HSN CODE │ ORDER #     │ ART #
64061020 │2606 - 407 │Fire Fighter GTX III U│Black │Boot Uppers  │ Cow         │         10   │   77.60      │      776.00
64061020 │2607 - 405 │ Oslo GTX 3.0 Carbon U│Black │Shoe Uppers  │ Cow         │         400   │  33.06     │     13224.00
                                      TOTAL:-                               │                               14000.00
                                      LESS  TRADE  DISCOUNT   0.25% IN EURO:-                 │                 35.00
                                      TOTAL:-                               │                               13965.00
                                      ADD  AIR FREIGHT   50% IN EURO:-                    │                    531.28
FOREIGN  AGENT   COMM   8% IN EURO:1120.00
Amount Chargeable                        │       G-Total    │   14,496.28
"""


def _totals_check(header, items, text):
    return _find(validate_totals(header, items, "export", invoice_text=text), "HEADER_LINE_TOTAL")


def test_total_reconciles_through_printed_discount_and_freight():
    items = [{"ItmTaxableVal": 776.0}, {"ItmTaxableVal": 13224.0}]
    c = _totals_check({"invoice_total": 14496.28}, items, _ADJUSTED_TOTALS_TEXT)
    assert c.status == "PASS"
    assert "TRADE DISCOUNT" in c.message and "FREIGHT" in c.message
    assert c.difference == 0


def test_discount_only_invoice_reconciles():
    # 58,525.71 less a 0.25% discount of 146.31 -> 58,379.40, no freight.
    text = "TOTAL:-   │   58525.71\nLESS  TRADE  DISCOUNT   0.25% IN EURO:-   │   146.31\n"
    c = _totals_check({"invoice_total": 58379.40}, [{"ItmTaxableVal": 58525.71}], text)
    assert c.status == "PASS"


def test_commission_line_is_never_added_to_the_total():
    # "FOREIGN AGENT COMM ... 1120.00" is informational, not part of the
    # total: the reconciling subset is discount + freight only.
    items = [{"ItmTaxableVal": 14000.0}]
    c = _totals_check({"invoice_total": 14496.28}, items, _ADJUSTED_TOTALS_TEXT)
    assert c.status == "PASS" and "COMM" not in c.message


def test_mismatch_the_printed_adjustments_cannot_explain_is_still_an_error():
    # Discount/freight lines exist, but the extracted total is off by more
    # than they account for -> a real problem, must NOT be waved through.
    items = [{"ItmTaxableVal": 14000.0}]
    # printed total LARGER than the lines can account for -> flagged as an
    # incomplete item list (still an ERROR, so it can't pass as valid)
    checks = validate_totals({"invoice_total": 15000.00}, items, "export", invoice_text=_ADJUSTED_TOTALS_TEXT)
    c = _find(checks, "LINE_ITEMS_INCOMPLETE")
    assert c.status == "ERROR" and _find(checks, "HEADER_LINE_TOTAL") is None


def test_adjustment_lines_without_amounts_do_not_explain_a_mismatch():
    # The real 6167 case: "LESS TRADE DISCOUNT 0.25%" is printed but its
    # amount isn't in the text. The percentage must not be mistaken for an
    # amount, so nothing reconciles and the mismatch stays flagged.
    text = "TOTAL:-\nLESS TRADE   DISCOUNT   0.25% IN EURO:-\nADD  AIR FREIGHT   50% IN EURO:-\n"
    checks = validate_totals({"invoice_total": 45386.09}, [{"ItmTaxableVal": 44698.0}], "export", invoice_text=text)
    assert _find(checks, "LINE_ITEMS_INCOMPLETE").status == "ERROR"


def test_mismatch_with_no_adjustment_lines_at_all_is_still_an_error():
    # The Piramal case: a genuinely wrong printed total, nothing in the
    # text that could explain it.
    items = [{"ItmTaxableVal": 51166.08}, {"ItmTaxableVal": 27216.00}]
    c = _totals_check({"invoice_total": 51166.08}, items, "Grand Total 51,166.08\n")
    assert c.status == "ERROR"


def test_without_source_text_behaviour_is_unchanged():
    items = [{"ItmTaxableVal": 14000.0}]
    c = _find(validate_totals({"invoice_total": 14496.28}, items, "export"), "LINE_ITEMS_INCOMPLETE")
    assert c.status == "ERROR"


def test_exact_match_never_needs_the_adjustment_lines():
    text = "LESS DISCOUNT 35.00\nADD FREIGHT 531.28\n"
    c = _totals_check({"invoice_total": 14000.0}, [{"ItmTaxableVal": 14000.0}], text)
    assert c.status == "PASS" and c.message == ""


def test_adjusted_total_passes_validation_end_to_end():
    result = _export_result(header_overrides={"invoice_total": 14496.28})
    item = result["sheets"]["ITEM"][0]
    item.update(Item_Qty=1, Item_Unit_Price=14000.0, ItmTaxableVal=14000.0)
    validation = validate_invoice(result, invoice_text=_ADJUSTED_TOTALS_TEXT, shipment_type="export")
    c = _find(validation.checks, "HEADER_LINE_TOTAL")
    assert c.status == "PASS"


# ── LINE_ITEMS_INCOMPLETE vs HEADER_LINE_TOTAL ──────────────────────────
# Confirmed real case: 4 JCB invoices whose source rows summed exactly to
# the printed total, but the model had extracted only the first page's rows.

def test_lines_short_of_the_printed_total_are_reported_as_incomplete_and_retryable():
    result = _export_result(header_overrides={"invoice_total": 10289.76})
    item = result["sheets"]["ITEM"][0]
    item.update(Item_Qty=1, Item_Unit_Price=5992.54, ItmTaxableVal=5992.54)
    validation = validate_invoice(result, invoice_text="", shipment_type="export")
    c = _find(validation.checks, "LINE_ITEMS_INCOMPLETE")
    assert c.status == "ERROR" and c in validation.retryable_errors
    assert "4297.22" in c.message and "EVERY page" in c.message
    assert _find(validation.checks, "HEADER_LINE_TOTAL") is None


def test_lines_over_the_printed_total_stay_a_non_retryable_header_mismatch():
    # The Piramal case: lines add up to MORE than the printed total. A retry
    # can only push the model toward the sum, so it must never be retryable.
    items = [{"ItmTaxableVal": 51166.08}, {"ItmTaxableVal": 27216.00}]
    checks = validate_totals({"invoice_total": 51166.08}, items, "export")
    assert _find(checks, "HEADER_LINE_TOTAL").status == "ERROR"
    assert _find(checks, "LINE_ITEMS_INCOMPLETE") is None
    result = _export_result(header_overrides={"invoice_total": 51166.08})
    result["sheets"]["ITEM"].append(dict(result["sheets"]["ITEM"][0], Item_Ser_No=2, ItmTaxableVal=27216.0, Item_Unit_Price=5.25, Item_Qty=5184))
    validation = validate_invoice(result, invoice_text="", shipment_type="export")
    assert not any(c.rule == "HEADER_LINE_TOTAL" for c in validation.retryable_errors)


def test_incomplete_item_list_feedback_goes_to_the_items_pass_only():
    result = _export_result(header_overrides={"invoice_total": 60000.00})  # fixture's one line is 51,166.08
    validation = validate_invoice(result, invoice_text="", shipment_type="export")
    header_fb, items_fb = build_retry_feedback(validation)
    assert header_fb is None and items_fb is not None and "LINE_ITEMS_INCOMPLETE" in items_fb


# ── HSN_FORMAT: Item No must not end up in the HSN column ───────────────

def _hsn_checks(value):
    items = [{"Item_RITC": value}]
    return [c for c in validate_format_and_types({}, items, "export") if c.rule == "HSN_FORMAT"]


def test_item_numbers_are_not_valid_hsn_codes():
    for bad in ("160", "10", "120", 250):  # the real values seen on the JCB invoices
        (c,) = _hsn_checks(bad)
        assert c.status == "ERROR" and c.retryable is True, bad
        assert "item or line number" in c.message


def test_real_hsn_codes_pass_in_any_printed_style():
    for ok in ("8481 80 90", "84314980", "8431.49.90", "4016 93 40", "8431", "843149"):
        assert _hsn_checks(ok)[0].status == "PASS", ok


def test_blank_hsn_is_never_checked():
    assert _hsn_checks("") == [] and _hsn_checks(None) == []


def test_unusual_length_hsn_is_only_a_warning():
    for odd in ("8431499012", "84314"):
        (c,) = _hsn_checks(odd)
        assert c.status == "WARNING" and not c.retryable, odd


def test_a_line_number_in_the_hsn_column_fails_validation_end_to_end():
    result = _export_result()
    result["sheets"]["ITEM"][0]["Item_RITC"] = "160"
    validation = validate_invoice(result, invoice_text="", shipment_type="export")
    assert not validation.valid
    assert any(c.rule == "HSN_FORMAT" for c in validation.retryable_errors)
    _, items_fb = build_retry_feedback(validation)
    assert items_fb and "HSN Code" in items_fb
