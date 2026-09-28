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
