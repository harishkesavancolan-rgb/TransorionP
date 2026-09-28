"""Smoke tests for detect_type.py's import/export/unknown classification."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detect_type import (
    _BUYER_LABELS,
    _SUPPLIER_LABELS,
    _COLUMN_BOUNDARY_MARKER as C,
    _label_positions,
    _label_proximity_hits,
    detect_shipment_type,
)


def test_gst_supply_type_expwp_detected_as_export():
    # Confirmed real case (1055044694.pdf, a Nokia GST e-invoice): no
    # label matching _SUPPLIER_LABELS appears anywhere -- the supplier's
    # own address block uses first-person framing ("Our Registered
    # Office Address"), not "Exporter:"/"Seller:" -- so the GST Supply
    # Type field ("GST   EXPWP   N   INV") is the ONLY structural signal
    # available. Must be picked up as a keyword hit.
    text = "Tax Scheme  Supply Type  Reverse Charge  Doc Type Code\nGST   EXPWP   N   INV\n"
    result = detect_shipment_type(text)
    assert result.shipment_type == "export"
    assert result.signal == "keyword"


def test_export_keywords_detected():
    text = "SHIPPING BILL NO. 1234567 dated 01-01-2026\nLET EXPORT ORDER granted\nExporter: ACME India Pvt Ltd"
    result = detect_shipment_type(text)
    assert result.shipment_type == "export"


def test_import_keywords_detected():
    text = "BILL OF ENTRY NO. 9988776 dated 01-01-2026\nInto Bond clearance\nImporter: ACME India Pvt Ltd"
    result = detect_shipment_type(text)
    assert result.shipment_type == "import"


def test_neutral_text_is_unknown():
    text = "Thank you for your business. Please remit payment within 30 days."
    result = detect_shipment_type(text)
    assert result.shipment_type == "unknown"


def test_structural_signal_corroborates_export():
    text = (
        "COMMERCIAL INVOICE\n"
        "Country of Origin: India\n"
        "Country of Destination: Germany\n"
        "Shipping Bill No. 55512345\n"
    )
    result = detect_shipment_type(text)
    assert result.shipment_type == "export"
    assert result.signal == "both"


# ── Column-bleed false positive (Reroute_CCU50 real case) ───────────────

def _two_column(*rows: tuple[str, str]) -> str:
    return "\n".join(f"        {left:<70}{C}{' ' * 70}{right}" for left, right in rows)


def test_supplier_side_ignores_buyer_column_india_on_shared_rows():
    # Confirmed real case: a UK supplier's own address (left column) sat
    # row-by-row next to the Indian buyer's address (right column). A
    # flat proximity window starting at "SELLER/ SUPPLIER:" used to read
    # straight across the "│" into the buyer's own "INDIA" a few rows
    # down, wrongly making the supplier side register an India hit too on
    # a genuine IMPORT invoice (UK supplier -> Indian buyer) -- with both
    # sides then showing a hit, detection came back "unknown" instead of
    # "import".
    text_lower = _two_column(
        ("seller/ supplier:", "zip code 700054"),
        ("hyve solutions europe limited:", "india"),
        ("technology park, telford", "gstn: 19aaoca5897l1zn"),
        ("shropshire tf3 3ah gb", ""),
    )
    buyer_starts = [s for s, _ in _label_positions(text_lower, _BUYER_LABELS)]
    supplier_starts = [s for s, _ in _label_positions(text_lower, _SUPPLIER_LABELS)]
    supplier_hits = _label_proximity_hits(text_lower, _SUPPLIER_LABELS, other_side_starts=buyer_starts)
    assert supplier_hits == 0


def test_buyer_side_still_finds_its_own_india_in_two_column_layout():
    text_lower = _two_column(
        ("seller/ supplier:", "bill to:"),
        ("hyve solutions europe limited:", "amazon data services india pvt ltd"),
        ("technology park, telford", "kolkata, west bengal"),
        ("shropshire tf3 3ah gb", "india"),
    )
    supplier_starts = [s for s, _ in _label_positions(text_lower, _SUPPLIER_LABELS)]
    buyer_starts = [s for s, _ in _label_positions(text_lower, _BUYER_LABELS)]
    buyer_hits = _label_proximity_hits(text_lower, _BUYER_LABELS, other_side_starts=supplier_starts)
    assert buyer_hits == 1


def test_single_column_multiline_address_still_matches():
    # Sanity: a normal multi-line address with no sibling column at all
    # must be unaffected by the column-restriction logic.
    text_lower = (
        "buyer:\n"
        "acme imports pvt ltd\n"
        "123 msg road\n"
        "chennai, tamil nadu\n"
        "india\n"
    )
    buyer_hits = _label_proximity_hits(text_lower, _BUYER_LABELS)
    assert buyer_hits == 1


def test_label_and_its_own_value_across_a_pipe_on_the_same_row_still_matches():
    # Confirmed real regression from an earlier, stricter version of the
    # column-bleed fix (INV7.pdf): "│" is also routinely used WITHIN a
    # single row as a plain label/value separator packing several
    # unrelated fields side by side -- "Ship To Messrs: │ Lenovo India
    # Pvt Ltd. │ FCR#". Restricting the label's own first line to its own
    # column cut it off from its own value in the very next cell, wrongly
    # dropping a correct India hit. The label's own line must stay
    # unrestricted.
    text_lower = f"        ship to messrs:{C}lenovo india pvt ltd.{C}fcr#\n"
    buyer_hits = _label_proximity_hits(text_lower, _BUYER_LABELS)
    assert buyer_hits == 1


def test_label_word_embedded_inside_an_unrelated_value_is_not_a_hit():
    # Confirmed real false positive: "seller" matched inside "FCA Seller'
    # Premises" -- an Incoterms phrase that is itself the VALUE of an
    # unrelated "DELIVERY TERMS:" field, not a party label. A proximity
    # scan anchored there read across a further "│" into a THIRD,
    # unrelated column's "...India..." on the same row. A genuine field
    # label always starts its own cell; "seller" here had "fca " in front
    # of it within the same cell, so it must not count as a label match.
    text_lower = (
        f"        delivery terms:{C}fca seller' premises{C}"
        f"stt global data centres india private limited\n"
    )
    supplier_hits = _label_proximity_hits(text_lower, _SUPPLIER_LABELS)
    assert supplier_hits == 0


# ── Pre-existing documented false-positive guards (regression coverage) ──

def test_ship_from_to_india_route_phrase_is_not_a_supplier_india_hit():
    # "Ship From: Changzhou,CN To India" names India as the DESTINATION of
    # the route, not the ship-from party's own location -- must not count
    # as a supplier-side India hit.
    text_lower = "ship from: changzhou,cn to india\n"
    supplier_hits = _label_proximity_hits(text_lower, _SUPPLIER_LABELS)
    assert supplier_hits == 0


def test_buyers_order_no_reference_field_is_not_treated_as_a_buyer_label():
    # "Buyer's Order No. & Date" names the OTHER party's paperwork, not
    # this section's own address block -- must be excluded from
    # _label_positions entirely (see _is_reference_field_label).
    text_lower = "buyer's order no. & date: po-84920 dated 01-01-2026\nsupplier: acme exports, chennai, india\n"
    positions = _label_positions(text_lower, _BUYER_LABELS)
    assert positions == []
