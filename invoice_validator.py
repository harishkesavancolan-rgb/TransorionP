"""
invoice_validator.py
---------------------
Deterministic business-rule / consistency validator for the FINAL,
template-shaped extraction result -- the same {"header": ..., "sheets":
{...}} shape main.py/api.py already build via validate.py's
validate_and_coerce().

This is a different concern from the existing validate.py:
  - validate.py maps the LLM's generic JSON onto real template columns and
    coerces/cleans values (currency strings -> numbers, GSTIN repair,
    state-code cross-check, ...). It runs ONCE, unconditionally, and never
    decides whether to retry anything.
  - invoice_validator.py checks whether the RESULT of that mapping is
    internally consistent -- arithmetic that should reconcile, fields that
    should agree with each other, fields that should agree with the source
    text -- and classifies every check as PASS / WARNING / ERROR /
    NOT_CHECKED. It never invents or corrects a value itself; that's
    reextraction.py's job, by asking the LLM to look again with feedback
    describing exactly what didn't reconcile.

Design principles (the "why" behind every rule below):
  - A rule only runs when every field it needs is actually present and
    non-empty. Missing inputs -> NOT_CHECKED, never a guessed ERROR --
    this project's invoices routinely have blank optional sections (see
    validate.py's own module docstring on the BOE sheet), and "field
    wasn't printed on this invoice" is not the same defect as "field was
    extracted wrong".
  - Only ERROR-level findings whose rule name is in RETRYABLE_RULES gate
    re-extraction (see reextraction.py). WARNING-level findings are
    informational and never retried.
  - Field names differ between the export ITEM sheet and the import BOE
    sheet (see templates_config.ITEM_SHEET_NAME). _ITEM_FIELDS below is
    the single place mapping a validation concept ("quantity", "unit
    price", ...) to the real column name per shipment type -- built by
    directly reading validate.py's _map_export_item / _map_import_item
    and schema.py's actual template columns, not guessed. Where a concept
    has no column on a template (BOE has no per-line monetary amount
    column at all), the mapping is None and every rule needing it
    correctly falls back to NOT_CHECKED for that shipment type.
  - Known structural note, worth being upfront about: several of the
    "cross-field" checks below (quantity vs tariff quantity, unit vs
    tariff unit, header state code vs item state code, header package
    count vs item package count) are, in the CURRENT validate.py mappers,
    populated by literally copying the same source value into both
    columns (e.g. _map_export_item sets both "Item_Qty" and "Qty Tariff"
    from the same `item["quantity"]`). That makes these specific checks a
    structural tautology today -- they cannot fail unless validate.py's
    mapping logic changes to extract the two sides independently. They're
    still implemented (the spec calls for them, and they become live
    safety nets the moment that mapping changes), but it would be
    dishonest to present them as independently-verified signals right now.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from validate import _GSTIN_RE, _is_valid_ad_code


# ── Configuration ───────────────────────────────────────────────────────

# Absolute monetary tolerance for every arithmetic reconciliation check
# (line qty*price vs printed amount, header total vs summed line amounts,
# tax amount vs taxable_value*rate, ...). A single configurable knob per
# the spec, rather than a different magic number hardcoded per rule.
AMOUNT_TOLERANCE = Decimal("0.05")

# Rule names that trigger re-extraction when they fail at ERROR severity
# (see reextraction.py). Kept as an explicit allow-list rather than
# "every ERROR retries automatically" so a new rule can be added at ERROR
# severity without silently becoming retry-worthy until that's a
# deliberate decision -- e.g. a future rule might be ERROR-severity but
# not worth spending a retry attempt on.
RETRYABLE_RULES = {
    "REQUIRED_FIELD_MISSING",
    "QTY_X_UNIT_PRICE",
    "HEADER_LINE_TOTAL",
    "TAX_RECONCILIATION",
    "PART_NUMBER_SOURCE_MATCH",
    "ORDER_NUMBER_SOURCE_MATCH",
    "MODEL_SOURCE_MATCH",
    "COUNTRY_OF_ORIGIN_CONSISTENCY",
    "FORMAT_TYPE",
}


# ── Result structures ───────────────────────────────────────────────────

@dataclass
class ValidationCheck:
    rule: str
    status: str  # "PASS" | "WARNING" | "ERROR" | "NOT_CHECKED"
    message: str = ""
    item: int | None = None  # None = header-level; else 1-based item index
    field: str | None = None
    expected: Any = None
    actual: Any = None
    difference: Any = None
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("expected", "actual", "difference"):
            if isinstance(d[k], Decimal):
                d[k] = float(d[k])
        return d


@dataclass
class ValidationResult:
    valid: bool
    status: str  # "VALID" | "INVALID"
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "ERROR"]

    @property
    def warnings(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "WARNING"]

    @property
    def passed_rules(self) -> list[str]:
        return [c.rule for c in self.checks if c.status == "PASS"]

    @property
    def retryable_errors(self) -> list[ValidationCheck]:
        return [c for c in self.errors if c.retryable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "errors": [c.to_dict() for c in self.errors],
            "warnings": [c.to_dict() for c in self.warnings],
            "checks": [c.to_dict() for c in self.checks],
            "passed_rules": self.passed_rules,
        }


# ── Field-name mapping: validation concept -> real template column ──────

# Built directly from validate.py's _map_export_item / _map_import_item
# and schema.py's actual template columns -- not guessed. BOE originally
# had no per-line monetary "amount" column at all; "Item_Taxable_Val" was
# added to the BOE sheet (templates/IMP_TEMPLET.xlsx) and wired up in
# validate.py's _map_import_item (source: the same per-line "Extended
# Price" the model already captures into the generic item's line_total --
# it was being extracted correctly all along, just never mapped anywhere)
# specifically so QTY_X_UNIT_PRICE and HEADER_LINE_TOTAL could actually run
# for import. BOE still has no per-item IGST rate/amount columns (only
# notification-NUMBER reference fields), so igst_amount/igst_rate stay
# None for import and TAX_RECONCILIATION stays NOT_CHECKED there -- see
# validate_taxes()'s own explicit shipment_type gate below, unaffected by
# this.
_ITEM_FIELDS: dict[str, dict[str, str | None]] = {
    "export": {
        "ser_no": "Item_Ser_No", "description": "Item_Desc",
        "quantity": "Item_Qty", "unit": "Item_Unit1",
        "unit_price": "Item_Unit_Price", "line_amount": "ItmTaxableVal",
        "tariff_qty": "Qty Tariff", "tariff_unit": "Unit Tariff",
        "country_of_origin": "Itm_Source_Cntry", "hsn": "Item_RITC",
        "igst_amount": "ItmIGSTamt", "igst_rate": "ItmIGSTPer",
        "state_code": "State Code", "total_packages": "ItmTotpkg",
    },
    "import": {
        "ser_no": "SL_No", "description": "Item_Desc1",
        "quantity": "Item_Qty", "unit": "Item_Unit",
        "unit_price": "Item_Unit_Price", "line_amount": "Item_Taxable_Val",
        "tariff_qty": "Qty_Tariff", "tariff_unit": "Unit_Tariff",
        "country_of_origin": "Item_Country_Org", "hsn": "Item_RITC",
        "igst_amount": None, "igst_rate": None,
        "state_code": None, "total_packages": None,
    },
}

# Present on every row regardless of shipment type -- validate.py's
# validate_and_coerce() always injects these onto the mapped row (see its
# mapped_rows loop), independent of which template applies.
_COMMON_ITEM_FIELDS = ("part_number", "model_or_type", "order_number")

_REQUIRED_ITEM_CONCEPTS = ("ser_no", "description", "quantity", "unit", "unit_price")


def _item_field(shipment_type: str, concept: str) -> str | None:
    return _ITEM_FIELDS.get(shipment_type, {}).get(concept)


# ── Numeric helpers ─────────────────────────────────────────────────────

def _to_decimal(value: Any) -> Decimal | None:
    """Best-effort Decimal conversion. None means "not a real number
    present" (missing/blank/non-numeric) -- callers must treat that as
    "can't check this", never as zero."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, float):
            # round-trip through str first to avoid binary-float noise
            # (e.g. 9.87 stored as a float can compare as 9.8699999...)
            return Decimal(str(value))
        return Decimal(value)
    except (InvalidOperation, ValueError, TypeError):
        return None


def _normalize_token(s: Any) -> str:
    """Uppercase, strip, collapse internal whitespace -- for comparing
    codes/numbers that may have incidental spacing differences between
    the source text and the extracted field, without discarding
    meaningful punctuation (hyphens, slashes) that part/order numbers
    routinely use."""
    return re.sub(r"\s+", "", str(s or "")).upper()


# ══════════════════════════════════════════════════════════════════════
# 1. Required / optional / conditional field checks
# ══════════════════════════════════════════════════════════════════════

def validate_required_fields(
    items: list[dict[str, Any]], shipment_type: str
) -> list[ValidationCheck]:
    """
    REQUIRED: ser_no, description, quantity, unit, unit_price -- a row
    without these isn't a usable line item on either template.
    CONDITIONAL (handled implicitly, not as a hardcoded list): a concept
    that has no column on this shipment type's template (e.g. BOE has no
    line-amount column) is simply never checked for it -- that's the
    schema-driven distinction the spec asks for, not a blanket "every
    field must be filled" pass over both templates' full column sets.
    OPTIONAL: everything else (model_or_type, ...) is never flagged here.
    """
    checks = []
    for idx, item in enumerate(items, start=1):
        for concept in _REQUIRED_ITEM_CONCEPTS:
            col = _item_field(shipment_type, concept)
            if col is None:
                continue
            val = item.get(col)
            if val is None or val == "":
                checks.append(ValidationCheck(
                    rule="REQUIRED_FIELD_MISSING", status="ERROR", item=idx,
                    field=col, retryable=True,
                    message=f"Item {idx}: required field '{col}' ({concept}) is missing.",
                ))
            else:
                checks.append(ValidationCheck(
                    rule="REQUIRED_FIELD_MISSING", status="PASS", item=idx, field=col,
                ))
    return checks


# ══════════════════════════════════════════════════════════════════════
# 2. Line-item arithmetic: Quantity x Unit Price = Line Amount
# ══════════════════════════════════════════════════════════════════════

def validate_line_items(
    items: list[dict[str, Any]], shipment_type: str, tolerance: Decimal = AMOUNT_TOLERANCE
) -> list[ValidationCheck]:
    """
    Quantity x Unit Price = Line Amount, per item, only where the
    template actually has a line-amount column to compare against (see
    _ITEM_FIELDS -- both export's ItmTaxableVal and import's
    Item_Taxable_Val qualify; a template with neither would fall back to
    NOT_CHECKED here, though that's no longer the case for either
    template today). Does NOT assume a field literally named
    "taxable_value" is always the line amount -- it looks up the concept
    through _ITEM_FIELDS, which is keyed off the real per-template column
    names.
    """
    checks = []
    qty_col = _item_field(shipment_type, "quantity")
    price_col = _item_field(shipment_type, "unit_price")
    amount_col = _item_field(shipment_type, "line_amount")

    for idx, item in enumerate(items, start=1):
        if amount_col is None:
            checks.append(ValidationCheck(
                rule="QTY_X_UNIT_PRICE", status="NOT_CHECKED", item=idx,
                message=(
                    f"Item {idx}: the {shipment_type} template has no per-line amount "
                    f"column to reconcile quantity x unit price against."
                ),
            ))
            continue

        qty = _to_decimal(item.get(qty_col))
        price = _to_decimal(item.get(price_col))
        amount = _to_decimal(item.get(amount_col))
        if qty is None or price is None or amount is None:
            checks.append(ValidationCheck(
                rule="QTY_X_UNIT_PRICE", status="NOT_CHECKED", item=idx,
                message=f"Item {idx}: quantity, unit price, or line amount is missing -- cannot check.",
            ))
            continue

        expected = (qty * price).quantize(Decimal("0.01"))
        actual = amount.quantize(Decimal("0.01")) if amount == amount.to_integral() else amount
        diff = abs(expected - amount)
        if diff <= tolerance:
            checks.append(ValidationCheck(
                rule="QTY_X_UNIT_PRICE", status="PASS", item=idx,
                field=amount_col, expected=expected, actual=amount, difference=diff,
            ))
        else:
            checks.append(ValidationCheck(
                rule="QTY_X_UNIT_PRICE", status="ERROR", item=idx, field=amount_col,
                expected=expected, actual=amount, difference=diff, retryable=True,
                message=(
                    f"Item {idx}: Quantity ({qty}) x Unit Price ({price}) = {expected}, "
                    f"but extracted {amount_col} is {amount} (difference {diff})."
                ),
            ))
    return checks


# ══════════════════════════════════════════════════════════════════════
# 3 & 4. Header total vs line totals, and other arithmetic relationships
# ══════════════════════════════════════════════════════════════════════

def _parse_header_total(header: dict[str, Any]) -> Decimal | None:
    """
    Parses header.invoice_total, which may legitimately be several numbers
    joined by " / " instead of one -- the same convention this pipeline's
    own prompt already documents for invoice_number/order_number on a PDF
    that combines multiple sub-invoices into one document (see
    llm_extract.py's _HEADER_SYSTEM_PROMPT). Confirmed on a real combined
    invoice: invoice_number was "IN2604006721 / IN2604006722" (two real
    sub-invoices) and invoice_total was, correctly, "10721.49 / 29812.90"
    -- the individual totals of the two sub-invoices -- but treating that
    as "not a number" rejected the right answer for a document that
    genuinely has no single total.

    Only parsed as combined when header.invoice_number ITSELF shows the
    same " / " pattern -- i.e. only when there's independent evidence
    this really is a combined document, not just because invoice_total
    happens to contain a slash for some unrelated (and possibly genuinely
    wrong) reason on an ordinary single invoice.
    """
    raw = header.get("invoice_total")
    if raw is None or raw == "":
        return None
    single = _to_decimal(raw)
    if single is not None:
        return single
    invoice_number = str(header.get("invoice_number") or "")
    if "/" not in invoice_number:
        return None
    parts = [p.strip() for p in str(raw).split("/")]
    decimals = [d for d in (_to_decimal(p) for p in parts) if d is not None]
    return sum(decimals, Decimal("0")) if decimals else None


def validate_totals(
    header: dict[str, Any], items: list[dict[str, Any]], shipment_type: str,
    tolerance: Decimal = AMOUNT_TOLERANCE,
) -> list[ValidationCheck]:
    """
    HEADER_LINE_TOTAL: calculated_total = sum(valid line amounts) vs
    header.invoice_total. Only runs when the template has a summable
    per-line amount column -- both export's ItmTaxableVal and import's
    Item_Taxable_Val qualify today; NOT_CHECKED only for a hypothetical
    template with neither.

    A line item missing its own amount is EXCLUDED from the sum rather
    than treated as zero: a genuinely-zero line and a not-extracted line
    aren't the same thing, and silently summing a missing value as 0
    would produce a false HEADER_LINE_TOTAL mismatch that's really a
    missing-data problem -- already caught separately by
    REQUIRED_FIELD_MISSING / QTY_X_UNIT_PRICE for that specific row.

    Other header-level arithmetic the spec asks for (subtotal vs sum of
    lines, discount amount vs discount percentage, round-off, freight,
    other charges) is deliberately NOT implemented as hardcoded checks
    here: this project's current header schema (see validate.py's
    _HEADER_FIELDS) has no subtotal / discount / round_off / freight /
    other_charges fields at all -- they are never extracted today, so a
    rule that "checks" them would either invent fields that don't exist
    or silently always NOT_CHECK, which is functionally the same as not
    having the rule. If/when those fields are added to the schema, the
    same _to_decimal + tolerance-comparison pattern used here extends
    directly to them -- see the module docstring's point on this.
    """
    checks = []
    amount_col = _item_field(shipment_type, "line_amount")
    invoice_total = _parse_header_total(header)

    if amount_col is None:
        checks.append(ValidationCheck(
            rule="HEADER_LINE_TOTAL", status="NOT_CHECKED",
            message=f"No per-line amount column on the {shipment_type} template to sum.",
        ))
        return checks

    line_amounts = [_to_decimal(it.get(amount_col)) for it in items]
    valid_amounts = [a for a in line_amounts if a is not None]
    if not valid_amounts or invoice_total is None:
        checks.append(ValidationCheck(
            rule="HEADER_LINE_TOTAL", status="NOT_CHECKED",
            message="Missing header invoice_total, or no line amounts available to sum.",
        ))
        return checks

    calculated_total = sum(valid_amounts, Decimal("0"))
    diff = abs(calculated_total - invoice_total)
    if diff <= tolerance:
        checks.append(ValidationCheck(
            rule="HEADER_LINE_TOTAL", status="PASS", field="invoice_total",
            expected=calculated_total, actual=invoice_total, difference=diff,
        ))
    else:
        # WARNING, not ERROR/retryable: a mismatch here is genuinely
        # ambiguous -- it means either the extraction misread the total,
        # OR the source document's own printed total simply doesn't equal
        # the sum of its own line items (confirmed real case: a Piramal
        # export invoice listed two items in its table, 51,166.08 and
        # 27,216.00, but its own printed "Grand Total" line -- and the
        # amount spelled out in words right next to it -- read only
        # 51,166.08, excluding the second item entirely; a genuine
        # arithmetic slip in the source PDF, not an extraction error).
        # There's no reliable way to tell those two cases apart from the
        # numbers alone, and retryable=True here can ONLY ever push the
        # model toward "matches the sum" -- so on the document-error case
        # it actively overwrites an already-correct value with a wrong
        # one (confirmed: that exact file passed attempt 1 with the
        # correct 51,166.08, got flagged ERROR, and "fixed" itself into
        # the wrong 78,382.08 on retry). A WARNING still surfaces the
        # mismatch for a human to check, without forcing a correction
        # that's as likely to be wrong as right.
        checks.append(ValidationCheck(
            rule="HEADER_LINE_TOTAL", status="WARNING", field="invoice_total",
            expected=calculated_total, actual=invoice_total, difference=diff,
            message=(
                f"Header invoice_total {invoice_total} does not match calculated "
                f"line total {calculated_total} (difference: {diff}). This may be a "
                f"genuine mismatch printed on the source document itself, not "
                f"necessarily an extraction error -- verify against the document."
            ),
        ))
    return checks


# ══════════════════════════════════════════════════════════════════════
# Tax reconciliation
# ══════════════════════════════════════════════════════════════════════

def validate_taxes(
    items: list[dict[str, Any]], shipment_type: str, tolerance: Decimal = AMOUNT_TOLERANCE
) -> list[ValidationCheck]:
    """
    Taxable Value x IGST Rate / 100 = IGST Amount, per item -- only
    checked for export, since that's the only template with these three
    as real, independently-populated columns (ItmTaxableVal, ItmIGSTPer,
    ItmIGSTamt; confirmed via schema.py). Export invoices in this
    pipeline are IGST-paid (per the invoice text: "SUPPLY AGAINST EXPORT
    IGST PAID INVOICE"), not CGST+SGST, which apply to intra-state
    domestic sales -- neither template actually has CGST/SGST columns
    (confirmed by direct inspection of both templates' schemas), so a
    CGST+SGST=Total GST rule is not hardcoded here with invented column
    names; it would always be NOT_CHECKED against fields that don't
    exist. BOE now has a taxable-value column (Item_Taxable_Val), but
    still no per-item IGST rate/amount columns -- its GST-related columns
    are all notification-NUMBER reference fields, not monetary amounts --
    so this specific three-way reconciliation still has no IGST rate or
    IGST amount to check against and stays NOT_CHECKED for import.
    """
    checks = []
    if shipment_type != "export":
        checks.append(ValidationCheck(
            rule="TAX_RECONCILIATION", status="NOT_CHECKED",
            message=f"No taxable-value/IGST-rate/IGST-amount columns on the {shipment_type} template.",
        ))
        return checks

    taxable_col = _item_field(shipment_type, "line_amount")
    rate_col = _item_field(shipment_type, "igst_rate")
    amount_col = _item_field(shipment_type, "igst_amount")

    for idx, item in enumerate(items, start=1):
        taxable = _to_decimal(item.get(taxable_col))
        rate_raw = item.get(rate_col)
        rate = _to_decimal(re.sub(r"[^\d.]", "", str(rate_raw))) if rate_raw not in (None, "") else None
        igst_amount = _to_decimal(item.get(amount_col))

        if taxable is None or rate is None or igst_amount is None:
            checks.append(ValidationCheck(
                rule="TAX_RECONCILIATION", status="NOT_CHECKED", item=idx,
                message=f"Item {idx}: taxable value, IGST rate, or IGST amount missing -- cannot check.",
            ))
            continue

        expected = (taxable * rate / Decimal("100")).quantize(Decimal("0.01"))
        diff = abs(expected - igst_amount)
        if diff <= tolerance:
            checks.append(ValidationCheck(
                rule="TAX_RECONCILIATION", status="PASS", item=idx, field=amount_col,
                expected=expected, actual=igst_amount, difference=diff,
            ))
        else:
            checks.append(ValidationCheck(
                rule="TAX_RECONCILIATION", status="ERROR", item=idx, field=amount_col,
                expected=expected, actual=igst_amount, difference=diff, retryable=True,
                message=(
                    f"Item {idx}: Taxable Value ({taxable}) x IGST Rate ({rate}%) = {expected}, "
                    f"but extracted {amount_col} is {igst_amount} (difference {diff})."
                ),
            ))
    return checks


# ══════════════════════════════════════════════════════════════════════
# 5. Cross-field consistency
# ══════════════════════════════════════════════════════════════════════

def validate_cross_fields(
    header: dict[str, Any], items: list[dict[str, Any]], shipment_type: str
) -> list[ValidationCheck]:
    """
    Cross-field checks the spec asks for. Worth knowing going in: some of
    these compare a header value against an item-level column that
    validate.py's current mapper populates BY COPYING the header/generic
    value onto every row (see this module's docstring) -- Qty Tariff and
    Unit Tariff duplicate the item's OWN quantity/unit under a second
    column name (genuinely per-item already, just a naming tautology),
    and State Code copies the header's state (correctly invariant -- an
    invoice has one supplier/exporter state, not one per line).

    Package count (ItmTotpkg) is DIFFERENT: validate.py's mapper now uses
    each item's own package count when the invoice prints one per item
    (confirmed real case: two items, 108 and 504 packages on different
    pallet ranges), falling back to a copy of the header's whole-shipment
    total only when an item never got its own count extracted. So this
    is no longer a guaranteed tautology -- see validate_cross_fields's
    PACKAGE_COUNT_CONSISTENCY check below, which reconciles by SUM when
    items have genuinely distinct values, and by per-row equality when
    they're all header-copy fallbacks (never blindly sums fallback copies,
    which would overcount by a factor of N -- the "do NOT blindly sum
    repeated fields" trap the spec warns about).
    """
    checks = []

    # Country of origin: header vs each item's own value. Only flagged
    # when the item has its OWN non-empty, genuinely different value --
    # an item whose own field was left blank already defaults to the
    # header's value inside validate.py's mapper, so comparing that case
    # here would just be comparing the header to itself.
    origin_col = _item_field(shipment_type, "country_of_origin")
    header_origin = _normalize_token(header.get("country_of_origin"))
    if origin_col and header_origin:
        for idx, item in enumerate(items, start=1):
            item_origin = _normalize_token(item.get(origin_col))
            if not item_origin:
                checks.append(ValidationCheck(
                    rule="COUNTRY_OF_ORIGIN_CONSISTENCY", status="NOT_CHECKED", item=idx,
                    message=f"Item {idx}: no item-level country of origin to compare.",
                ))
            elif item_origin == header_origin:
                checks.append(ValidationCheck(
                    rule="COUNTRY_OF_ORIGIN_CONSISTENCY", status="PASS", item=idx,
                    field=origin_col, expected=header_origin, actual=item_origin,
                ))
            else:
                checks.append(ValidationCheck(
                    rule="COUNTRY_OF_ORIGIN_CONSISTENCY", status="WARNING", item=idx,
                    field=origin_col, expected=header_origin, actual=item_origin,
                    message=(
                        f"Item {idx}: country of origin '{item_origin}' differs from "
                        f"header country_of_origin '{header_origin}'."
                    ),
                ))

    # Item quantity vs tariff quantity (same document-level quantity
    # concept per the template's own column aliases -- see schema.py's
    # "Qty Tariff": ["Quantity for Tariff", "Tariff Quantity"]).
    qty_col = _item_field(shipment_type, "quantity")
    tariff_qty_col = _item_field(shipment_type, "tariff_qty")
    if qty_col and tariff_qty_col:
        for idx, item in enumerate(items, start=1):
            qty = _to_decimal(item.get(qty_col))
            tqty = _to_decimal(item.get(tariff_qty_col))
            if qty is None or tqty is None:
                checks.append(ValidationCheck(
                    rule="QTY_VS_TARIFF_QTY", status="NOT_CHECKED", item=idx,
                    message=f"Item {idx}: quantity or tariff quantity missing -- cannot check.",
                ))
            elif qty == tqty:
                checks.append(ValidationCheck(
                    rule="QTY_VS_TARIFF_QTY", status="PASS", item=idx,
                    field=tariff_qty_col, expected=qty, actual=tqty,
                ))
            else:
                checks.append(ValidationCheck(
                    rule="QTY_VS_TARIFF_QTY", status="ERROR", item=idx, field=tariff_qty_col,
                    expected=qty, actual=tqty, difference=abs(qty - tqty), retryable=True,
                    message=f"Item {idx}: quantity ({qty}) does not match tariff quantity ({tqty}).",
                ))

    # Item unit vs tariff unit (same document-level unit concept).
    unit_col = _item_field(shipment_type, "unit")
    tariff_unit_col = _item_field(shipment_type, "tariff_unit")
    if unit_col and tariff_unit_col:
        for idx, item in enumerate(items, start=1):
            unit = _normalize_token(item.get(unit_col))
            tunit = _normalize_token(item.get(tariff_unit_col))
            if not unit or not tunit:
                checks.append(ValidationCheck(
                    rule="UNIT_VS_TARIFF_UNIT", status="NOT_CHECKED", item=idx,
                    message=f"Item {idx}: unit or tariff unit missing -- cannot check.",
                ))
            elif unit == tunit:
                checks.append(ValidationCheck(
                    rule="UNIT_VS_TARIFF_UNIT", status="PASS", item=idx, field=tariff_unit_col,
                ))
            else:
                checks.append(ValidationCheck(
                    rule="UNIT_VS_TARIFF_UNIT", status="WARNING", item=idx, field=tariff_unit_col,
                    expected=unit, actual=tunit,
                    message=f"Item {idx}: unit ('{unit}') does not match tariff unit ('{tunit}').",
                ))

    # Header state code vs item state code (export only -- import's BOE
    # has no per-item state-code column).
    state_col = _item_field(shipment_type, "state_code")
    header_state = _normalize_token(header.get("state_code"))
    if state_col and header_state:
        for idx, item in enumerate(items, start=1):
            item_state = _normalize_token(item.get(state_col))
            if not item_state:
                checks.append(ValidationCheck(
                    rule="STATE_CODE_CONSISTENCY", status="NOT_CHECKED", item=idx,
                    message=f"Item {idx}: no item-level state code to compare.",
                ))
            elif item_state.lstrip("0") == header_state.lstrip("0"):
                checks.append(ValidationCheck(
                    rule="STATE_CODE_CONSISTENCY", status="PASS", item=idx, field=state_col,
                ))
            else:
                checks.append(ValidationCheck(
                    rule="STATE_CODE_CONSISTENCY", status="WARNING", item=idx, field=state_col,
                    expected=header_state, actual=item_state,
                    message=(
                        f"Item {idx}: state code '{item_state}' differs from "
                        f"header state_code '{header_state}'."
                    ),
                ))

    # Header package count vs item package info. validate.py's mapper now
    # populates each item's own package count when the invoice prints one
    # per item (confirmed real case: two items, 108 and 504 packages on
    # different pallet ranges), falling back to a copy of the header's
    # whole-shipment total only when an item never got its own count. That
    # means ItmTotpkg is no longer guaranteed identical across rows, so
    # this checks the relationship that's actually true either way: SUM
    # the item values and compare to the header total (the correct check
    # when each item genuinely has its own count, e.g. 108 + 504 = 612).
    # If that doesn't match, also accept the case every row is a plain
    # header-copy fallback: every item shares one identical value AND that
    # value already equals the header total. Only something outside BOTH
    # of those is flagged.
    pkg_col = _item_field(shipment_type, "total_packages")
    header_pkgs = _to_decimal(header.get("total_packages"))
    if pkg_col and header_pkgs is not None:
        item_pkgs_list = [(idx, _to_decimal(item.get(pkg_col))) for idx, item in enumerate(items, start=1)]
        checked = [(idx, v) for idx, v in item_pkgs_list if v is not None]
        if checked:
            values = [v for _, v in checked]
            summed = sum(values, Decimal("0"))
            all_same_as_header = all(v == header_pkgs for v in values)
            if summed == header_pkgs or all_same_as_header:
                checks.append(ValidationCheck(
                    rule="PACKAGE_COUNT_CONSISTENCY", status="PASS",
                    field="total_packages", expected=header_pkgs,
                    message="Item-level package counts reconcile with header.total_packages.",
                ))
            else:
                checks.append(ValidationCheck(
                    rule="PACKAGE_COUNT_CONSISTENCY", status="WARNING",
                    field="total_packages", expected=header_pkgs, actual=summed,
                    message=(
                        f"Item-level package counts sum to {summed}, which matches neither "
                        f"header.total_packages ({header_pkgs}) nor a per-row copy of it."
                    ),
                ))
        else:
            checks.append(ValidationCheck(
                rule="PACKAGE_COUNT_CONSISTENCY", status="NOT_CHECKED",
                message="No item-level package count populated to compare against the header.",
            ))

    return checks


# ══════════════════════════════════════════════════════════════════════
# 6 & 9. Description / source-text <-> extracted field validation
# ══════════════════════════════════════════════════════════════════════

# Deterministic label patterns for values that are commonly embedded in a
# line item's own description text, or printed near it. Each captures the
# code/number that follows the label -- matching is exact-substring after
# normalization (_normalize_token), never fuzzy/LLM-based.
_PART_NO_PATTERNS = [
    re.compile(r'P\s*/\s*N\.?\s*[:\-]?\s*([A-Za-z0-9\-\./]{3,})', re.IGNORECASE),
    # Colon/dash required here (unlike "P/N", which is distinctive enough
    # on its own): "Part No" without one is one edit away from false-
    # positive-matching ordinary words starting with "No" right after
    # "Part" (e.g. "PART NOMINAL VALUE" would otherwise read as
    # "Part No: MINAL VALUE").
    re.compile(r'Part\s*No\.?\s*[:\-]\s*([A-Za-z0-9\-\./]{3,})', re.IGNORECASE),
]
_MODEL_PATTERNS = [
    # The colon/dash separator is REQUIRED, not optional -- an earlier
    # version made it optional and "PLAIN ITEM WITH NO MODEL LABEL"
    # matched "LABEL" as a fake model value (caught by this module's own
    # test suite). A bare "Model" followed by an unrelated word with no
    # punctuation is not a labeled value; a real label always has one.
    re.compile(r'Model(?:\s*(?:No\.?|Number))?\s*[:\-]\s*([A-Za-z0-9\-\./]{2,})', re.IGNORECASE),
    re.compile(r'\bType\s*[:\-]\s*([A-Za-z0-9\-\./]{2,})', re.IGNORECASE),
]
_ORDER_NO_PATTERNS = [
    # Both require a mandatory "/" and mandatory "[:\-]" respectively --
    # an earlier, looser version of this pattern matched "STAL" out of
    # the ordinary word "POSTAL" (P + optional "/" + O + captured
    # trailing letters), caught by this module's own test suite.
    re.compile(r'P\s*/\s*O\.?\s*[:\-]?\s*([A-Za-z0-9\-]{3,})', re.IGNORECASE),
    re.compile(r"(?:Purchase\s*)?Order\s*No\.?\s*[:\-]\s*([A-Za-z0-9\-]{3,})", re.IGNORECASE),
]


def _find_labeled_value(patterns: list[re.Pattern], text: str) -> str | None:
    for pat in patterns:
        m = pat.search(text or "")
        if m:
            return m.group(1)
    return None


def validate_source_consistency(
    items: list[dict[str, Any]], shipment_type: str, invoice_text: str
) -> list[ValidationCheck]:
    """
    Catches LLM extraction mistakes specifically -- a value the model
    invented or mis-copied even though the source text says something
    else. Deterministic label matching (regex substring search, not
    fuzzy/LLM matching): if the item's own description (or, for
    order_number, the whole invoice text -- order numbers are usually in
    a separate column, not embedded in the description) contains a
    recognizable "P/N:", "Model:", or "P/O:"-style label, the labeled
    value is compared against the corresponding extracted field.

    Only ever produces PASS/ERROR/NOT_CHECKED here, never a WARNING for
    "field is empty but no label was found either" -- a description with
    no such label simply isn't evidence of anything, so it's
    NOT_CHECKED, not a missing-field complaint (that's
    validate_required_fields' job for genuinely required fields; model_or_type
    is optional and a bare absence of a "Model:" label proves nothing).
    """
    checks = []
    desc_col = _item_field(shipment_type, "description")

    for idx, item in enumerate(items, start=1):
        description = str(item.get(desc_col) or "")

        part_no_in_text = _find_labeled_value(_PART_NO_PATTERNS, description)
        extracted_part_no = _normalize_token(item.get("part_number"))
        if part_no_in_text is None:
            checks.append(ValidationCheck(
                rule="PART_NUMBER_SOURCE_MATCH", status="NOT_CHECKED", item=idx,
                message=f"Item {idx}: no 'P/N:'/'Part No:' label found in the description to check against.",
            ))
        elif _normalize_token(part_no_in_text) == extracted_part_no:
            checks.append(ValidationCheck(
                rule="PART_NUMBER_SOURCE_MATCH", status="PASS", item=idx, field="part_number",
                expected=part_no_in_text, actual=item.get("part_number"),
            ))
        elif not extracted_part_no:
            checks.append(ValidationCheck(
                rule="PART_NUMBER_SOURCE_MATCH", status="WARNING", item=idx, field="part_number",
                expected=part_no_in_text, actual="",
                message=f"Item {idx}: description names part number '{part_no_in_text}' but part_number was left empty.",
            ))
        else:
            checks.append(ValidationCheck(
                rule="PART_NUMBER_SOURCE_MATCH", status="ERROR", item=idx, field="part_number",
                expected=part_no_in_text, actual=item.get("part_number"), retryable=True,
                message=(
                    f"Item {idx}: description says part number '{part_no_in_text}', "
                    f"but extracted part_number is '{item.get('part_number')}'."
                ),
            ))

        model_in_text = _find_labeled_value(_MODEL_PATTERNS, description)
        extracted_model = _normalize_token(item.get("model_or_type"))
        if model_in_text is None:
            checks.append(ValidationCheck(
                rule="MODEL_SOURCE_MATCH", status="NOT_CHECKED", item=idx,
                message=f"Item {idx}: no 'Model:'/'Type:' label found in the description to check against.",
            ))
        elif _normalize_token(model_in_text) == extracted_model:
            checks.append(ValidationCheck(
                rule="MODEL_SOURCE_MATCH", status="PASS", item=idx, field="model_or_type",
            ))
        elif not extracted_model:
            checks.append(ValidationCheck(
                rule="MODEL_SOURCE_MATCH", status="WARNING", item=idx, field="model_or_type",
                expected=model_in_text, actual="",
                message=f"Item {idx}: description names model '{model_in_text}' but model_or_type was left empty.",
            ))
            # Deliberately not ERROR/retryable: model_or_type is an
            # OPTIONAL field per the spec's own classification, so a
            # missed one is worth surfacing but not worth spending a
            # retry attempt on.
        else:
            checks.append(ValidationCheck(
                rule="MODEL_SOURCE_MATCH", status="ERROR", item=idx, field="model_or_type",
                expected=model_in_text, actual=item.get("model_or_type"), retryable=True,
                message=(
                    f"Item {idx}: description says model '{model_in_text}', "
                    f"but extracted model_or_type is '{item.get('model_or_type')}'."
                ),
            ))

    # order_number: checked against the whole invoice text once per
    # distinct extracted value (not once per item) -- a single PO/order
    # reference is typically stated once and shared by several rows, so
    # checking it per-row against the same whole-document text would
    # just repeat the identical result N times.
    seen_order_values: set[str] = set()
    for idx, item in enumerate(items, start=1):
        extracted_order = str(item.get("order_number") or "")
        norm_order = _normalize_token(extracted_order)
        if not norm_order or norm_order in seen_order_values:
            continue
        seen_order_values.add(norm_order)

        order_in_text = _find_labeled_value(_ORDER_NO_PATTERNS, invoice_text)
        if order_in_text is None:
            checks.append(ValidationCheck(
                rule="ORDER_NUMBER_SOURCE_MATCH", status="NOT_CHECKED", item=idx,
                message="No 'P/O:'/'Order No:' label found in the source text to check against.",
            ))
        elif _normalize_token(order_in_text) in norm_order or norm_order in _normalize_token(order_in_text):
            # order_number sometimes legitimately carries extra context
            # (a date, a second PO) beyond the bare number found by the
            # label regex -- see llm_extract.py's own prompt rule 8 on
            # this -- so containment either direction counts as a match,
            # not strict equality.
            checks.append(ValidationCheck(
                rule="ORDER_NUMBER_SOURCE_MATCH", status="PASS", item=idx, field="order_number",
                expected=order_in_text, actual=extracted_order,
            ))
        else:
            checks.append(ValidationCheck(
                rule="ORDER_NUMBER_SOURCE_MATCH", status="ERROR", item=idx, field="order_number",
                expected=order_in_text, actual=extracted_order, retryable=True,
                message=(
                    f"Item {idx}: source text shows order number '{order_in_text}', "
                    f"but extracted order_number is '{extracted_order}'."
                ),
            ))

    return checks


# ══════════════════════════════════════════════════════════════════════
# 8. Format / type validation
# ══════════════════════════════════════════════════════════════════════

# Not exhaustive -- common invoice currencies only. An unrecognized but
# plausible-looking 3-letter code is a WARNING, never an ERROR, per the
# spec's "don't reject unusual but legitimate values without a clear
# rule" -- ISO 4217 has ~180 codes and this project has no need (or
# reliable way) to enumerate all of them.
_KNOWN_CURRENCIES = {
    "USD", "EUR", "GBP", "INR", "JPY", "CNY", "AUD", "CAD", "CHF", "SGD",
    "AED", "SAR", "HKD", "KRW", "SEK", "NOK", "DKK", "ZAR", "MXN", "BRL",
}

_DATE_FORMATS = (
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%d %b %Y",
    "%d.%m.%Y", "%Y/%m/%d",
)


def _parse_date_loose(s: str) -> datetime | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def validate_format_and_types(
    header: dict[str, Any], items: list[dict[str, Any]], shipment_type: str
) -> list[ValidationCheck]:
    """
    Structural/format checks -- numeric-ness, sign, date parseability,
    and the two identifier formats this project already has a rigorous,
    confirmed shape for (GSTIN, AD Code -- reusing validate.py's own
    _GSTIN_RE / _is_valid_ad_code rather than redefining them). IEC is
    checked far more leniently (WARNING, not ERROR) since this project
    has only one confirmed real example (10 digits) to go on, not a
    verified spec the way GSTIN's 15-character shape is documented and
    tested against multiple real invoices.
    """
    checks = []

    # -- Line items: quantity / unit_price numeric and non-negative --
    qty_col = _item_field(shipment_type, "quantity")
    price_col = _item_field(shipment_type, "unit_price")
    for idx, item in enumerate(items, start=1):
        for concept, col in (("quantity", qty_col), ("unit_price", price_col)):
            if col is None:
                continue
            raw = item.get(col)
            if raw is None or raw == "":
                continue  # absence is REQUIRED_FIELD_MISSING's concern, not FORMAT_TYPE's
            val = _to_decimal(raw)
            if val is None:
                checks.append(ValidationCheck(
                    rule="FORMAT_TYPE", status="ERROR", item=idx, field=col,
                    actual=raw, retryable=True,
                    message=f"Item {idx}: {concept} ('{raw}') is not numeric.",
                ))
            elif val < 0:
                checks.append(ValidationCheck(
                    rule="FORMAT_TYPE", status="ERROR", item=idx, field=col,
                    actual=val, retryable=True,
                    message=f"Item {idx}: {concept} ({val}) is negative.",
                ))
            else:
                checks.append(ValidationCheck(rule="FORMAT_TYPE", status="PASS", item=idx, field=col))

    # -- Header: invoice_total numeric and non-negative --
    # Uses _parse_header_total, not a plain numeric parse, so a
    # legitimately combined multi-sub-invoice total ("10721.49 /
    # 29812.90", matched to invoice_number "A / B") isn't flagged as
    # non-numeric -- see that helper's docstring.
    total_raw = header.get("invoice_total")
    if total_raw not in (None, ""):
        total_val = _parse_header_total(header)
        if total_val is None:
            checks.append(ValidationCheck(
                rule="FORMAT_TYPE", status="ERROR", field="invoice_total", actual=total_raw,
                retryable=True, message=f"invoice_total ('{total_raw}') is not numeric.",
            ))
        elif total_val < 0:
            checks.append(ValidationCheck(
                rule="FORMAT_TYPE", status="ERROR", field="invoice_total", actual=total_val,
                retryable=True, message=f"invoice_total ({total_val}) is negative.",
            ))
        else:
            checks.append(ValidationCheck(rule="FORMAT_TYPE", status="PASS", field="invoice_total"))

    # -- Header: invoice_number non-empty --
    if not str(header.get("invoice_number") or "").strip():
        checks.append(ValidationCheck(
            rule="REQUIRED_FIELD_MISSING", status="ERROR", field="invoice_number", retryable=True,
            message="invoice_number is empty.",
        ))
    else:
        checks.append(ValidationCheck(rule="REQUIRED_FIELD_MISSING", status="PASS", field="invoice_number"))

    # -- Header: invoice_date valid --
    date_raw = str(header.get("invoice_date") or "").strip()
    if date_raw:
        if _parse_date_loose(date_raw) is None:
            checks.append(ValidationCheck(
                rule="FORMAT_TYPE", status="WARNING", field="invoice_date", actual=date_raw,
                message=f"invoice_date '{date_raw}' doesn't match any recognized date format.",
            ))
        else:
            checks.append(ValidationCheck(rule="FORMAT_TYPE", status="PASS", field="invoice_date"))
    else:
        checks.append(ValidationCheck(
            rule="FORMAT_TYPE", status="NOT_CHECKED", field="invoice_date",
            message="invoice_date is empty.",
        ))

    # -- Header: currency --
    currency = str(header.get("currency") or "").strip().upper()
    if currency:
        if currency in _KNOWN_CURRENCIES:
            checks.append(ValidationCheck(rule="FORMAT_TYPE", status="PASS", field="currency"))
        elif re.match(r"^[A-Z]{3}$", currency):
            checks.append(ValidationCheck(
                rule="FORMAT_TYPE", status="WARNING", field="currency", actual=currency,
                message=f"currency '{currency}' looks like a 3-letter code but isn't in the common-currency list.",
            ))
        else:
            checks.append(ValidationCheck(
                rule="FORMAT_TYPE", status="ERROR", field="currency", actual=currency, retryable=True,
                message=f"currency '{currency}' doesn't look like a valid ISO currency code.",
            ))

    # -- Header: GSTIN shape (reuses validate.py's own regex) --
    # Only meaningful on export: there, "supplier" is the Indian exporter,
    # so supplier_tax_id is a real 15-char GSTIN. On import, "supplier" is
    # the foreign seller and this same field holds THEIR country's own
    # tax-ID format (VAT, EIN, ...), which has no fixed shape to check --
    # applying the GSTIN shape there produced false WARNINGs on genuinely
    # correct foreign VAT numbers, so it's NOT_CHECKED instead.
    gstin = str(header.get("supplier_tax_id") or "").strip()
    if gstin and shipment_type != "export":
        checks.append(ValidationCheck(
            rule="FORMAT_TYPE", status="NOT_CHECKED", field="supplier_tax_id",
            message="supplier_tax_id on an import invoice is the foreign supplier's own "
                    "tax ID (VAT/EIN/...), not an Indian GSTIN -- shape not checked.",
        ))
    elif gstin:
        if _GSTIN_RE.match(gstin):
            checks.append(ValidationCheck(rule="FORMAT_TYPE", status="PASS", field="supplier_tax_id"))
        else:
            checks.append(ValidationCheck(
                rule="FORMAT_TYPE", status="WARNING", field="supplier_tax_id", actual=gstin,
                message=f"supplier_tax_id '{gstin}' doesn't match the standard 15-character GSTIN shape.",
            ))
            # WARNING, not ERROR: validate.py's _repair_gstin already
            # attempted a safe repair before this ever runs, and already
            # emits its own warning for the unrepairable cases (a
            # doubled letter, a missing character) where guessing further
            # here would be no more reliable -- see that function's
            # docstring for why those are deliberately left alone.

    # -- Header: AD Code shape (reuses validate.py's own check) --
    ad_code = str(header.get("ad_code") or "").strip()
    if ad_code and not _is_valid_ad_code(ad_code):
        checks.append(ValidationCheck(
            rule="FORMAT_TYPE", status="WARNING", field="ad_code", actual=ad_code,
            message=f"ad_code '{ad_code}' doesn't match the expected 14-digit AD Code shape.",
        ))
    elif ad_code:
        checks.append(ValidationCheck(rule="FORMAT_TYPE", status="PASS", field="ad_code"))

    # -- Header: IEC -- lenient (WARNING only): one confirmed real shape
    # (10 characters) isn't enough evidence to hard-fail on.
    iec = str(header.get("iec_number") or "").strip()
    if iec and not re.match(r"^[A-Z0-9]{10}$", iec.upper()):
        checks.append(ValidationCheck(
            rule="FORMAT_TYPE", status="WARNING", field="iec_number", actual=iec,
            message=f"iec_number '{iec}' isn't the usual 10-character IEC shape.",
        ))
    elif iec:
        checks.append(ValidationCheck(rule="FORMAT_TYPE", status="PASS", field="iec_number"))

    return checks


# ══════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════

def validate_invoice(
    result: dict[str, Any],
    invoice_text: str,
    shipment_type: str,
    tolerance: Decimal = AMOUNT_TOLERANCE,
) -> ValidationResult:
    """
    Runs every check above against one extraction result (the same
    {"header": ..., "sheets": {...}} shape main.py/api.py produce) and
    returns a single aggregated ValidationResult.

    `result["valid"]` is False if and only if at least one ERROR-severity
    check fired -- WARNINGs and NOT_CHECKED never affect validity, per
    the spec ("Only ERROR-level validation failures should normally
    trigger re-extraction").
    """
    from templates_config import ITEM_SHEET_NAME

    header = result.get("header", {}) or {}
    item_sheet = ITEM_SHEET_NAME.get(shipment_type)
    items = (result.get("sheets", {}) or {}).get(item_sheet, []) or []

    checks: list[ValidationCheck] = []
    checks += validate_required_fields(items, shipment_type)
    checks += validate_line_items(items, shipment_type, tolerance)
    checks += validate_totals(header, items, shipment_type, tolerance)
    checks += validate_taxes(items, shipment_type, tolerance)
    checks += validate_cross_fields(header, items, shipment_type)
    checks += validate_source_consistency(items, shipment_type, invoice_text)
    checks += validate_format_and_types(header, items, shipment_type)

    for c in checks:
        if c.status == "ERROR" and c.rule in RETRYABLE_RULES:
            c.retryable = True

    has_error = any(c.status == "ERROR" for c in checks)
    return ValidationResult(valid=not has_error, status="INVALID" if has_error else "VALID", checks=checks)


# ══════════════════════════════════════════════════════════════════════
# Retry-feedback construction
# ══════════════════════════════════════════════════════════════════════

# Which rules are about the header vs. about line items -- used to route
# feedback to the header-extraction pass, the line-items pass, or both,
# so a re-extraction attempt only re-runs (and re-spends tokens on) the
# LLM pass that actually needs correcting. See reextraction.py.
_HEADER_RULES = {"HEADER_LINE_TOTAL"}
_ITEM_RULES = {
    "QTY_X_UNIT_PRICE", "TAX_RECONCILIATION", "PART_NUMBER_SOURCE_MATCH",
    "MODEL_SOURCE_MATCH", "ORDER_NUMBER_SOURCE_MATCH", "QTY_VS_TARIFF_QTY",
    "UNIT_VS_TARIFF_UNIT",
}
# REQUIRED_FIELD_MISSING, FORMAT_TYPE, COUNTRY_OF_ORIGIN_CONSISTENCY can
# legitimately be either, depending on which field failed -- routed per
# check below via `item is None`.


def _format_check_for_feedback(c: ValidationCheck) -> str:
    loc = f"ITEM {c.item}" if c.item is not None else "HEADER"
    parts = [f"{loc}: [{c.rule}] {c.message}"]
    if c.expected is not None or c.actual is not None:
        # str(), not repr() -- a raw Decimal('78382.08') / Decimal object
        # repr is noise the model has to parse around for no benefit;
        # plain "78382.08" is unambiguous and matches how the number
        # would actually appear if quoted back from the source invoice.
        parts.append(f"    expected: {c.expected!s}   extracted: {c.actual!s}")
    return "\n".join(parts)


def build_retry_feedback(
    validation: ValidationResult,
) -> tuple[str | None, str | None]:
    """
    Builds the compact, structured feedback text sent back to the LLM on
    a failed validation attempt (spec sections 10/11) -- returns
    (header_feedback, items_feedback), either of which is None if that
    pass had no failing checks routed to it, so reextraction.py can skip
    re-calling that pass entirely and reuse its previous, already-valid
    output.

    Only retryable ERRORs are included -- WARNINGs and non-retryable
    findings are noise for a re-extraction prompt (the model can't
    usefully act on "this is unusual but not necessarily wrong").
    """
    errors = validation.retryable_errors
    if not errors:
        return None, None

    header_errors = [c for c in errors if c.rule in _HEADER_RULES or (c.item is None and c.rule not in _ITEM_RULES)]
    item_errors = [c for c in errors if c.rule in _ITEM_RULES or (c.item is not None and c.rule not in _HEADER_RULES)]

    def _build(errs: list[ValidationCheck]) -> str | None:
        if not errs:
            return None
        lines = [
            "VALIDATION FAILED", "",
            f"{len(errs)} issue(s) found in your previous extraction:",
        ]
        for i, c in enumerate(errs, start=1):
            lines.append(f"{i}. {_format_check_for_feedback(c)}")
        lines += [
            "",
            "Re-check the original invoice text above and correct ONLY the fields "
            "named in these issues. Do not invent missing information. Do not modify "
            "any other field that was not flagged. Return the complete corrected JSON "
            "in the same format as before.",
        ]
        return "\n".join(lines)

    return _build(header_errors), _build(item_errors)
