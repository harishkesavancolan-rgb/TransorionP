"""
validate.py
------------
Turns the LLM's two generic payloads (invoice header + line items, from
llm_extract.py) into rows shaped for whichever customs template applies
(export ITEM sheet vs. import BOE sheet — see detect_type.py for how that's
decided).

Pipeline, in order:
1. `_clean_invoice_header()` — normalizes header fields (state code,
   country codes, currency-formatted numbers, RoDTEP Y/N).
2. `_clean_generic_item()` — per line item: drops subtotal/deduction rows,
   coerces qty/price/total to numbers, cross-checks qty*price against the
   printed total, resolves part_number/model_or_type from whichever
   alternate key the LLM used, and untangles FTA-code vs EU-end-use-code
   vs AD-code (a bank code, not a tariff code, despite the name).
3. `_map_export_item()` / `_map_import_item()` — the ONLY two places that
   know an ITEM-sheet or BOE-sheet column name. Each takes the same
   cleaned generic item + cleaned header and returns just the columns it
   can actually populate; every column the template defines that isn't
   set here comes back as "" via `validate_row_against_model()` (models.py)
   — never a guessed default, never a confidence score.
4. `_apply_export_domain_rules()` / `_apply_import_domain_rules()` —
   India-customs-specific cleanup that only makes sense once the row is in
   its final template shape (e.g. telling a container number apart from an
   airway bill number is an ITEM-sheet-only concept).

Note on fuzzy column-name matching: earlier versions of this pipeline had
the LLM emit template column names directly and fuzzy-matched whatever it
invented back onto real columns. The two-pass prompts in llm_extract.py
no longer do that -- they return a small fixed set of generic keys, and
step 3 above maps those to real column names deterministically. There is
nothing left in this design for fuzzy name-matching to rescue, so it isn't
here; a schema-driven single-prompt design would need it back.
"""
from __future__ import annotations
import re
from typing import Any

from models import build_sheet_models, validate_row_against_model
from templates_config import ITEM_SHEET_NAME


# ── Value unwrapping ───────────────────────────────────────────────────

def _unwrap_value(raw_val: Any) -> Any:
    """
    Both prompts explicitly forbid the {"value": ..., "confidence": ...}
    wrapper format, but this stays as a cheap defensive unwrap in case a
    model ignores that instruction -- without it, a stray wrapped value
    would land in the output as a raw dict instead of the scalar it holds.
    """
    if isinstance(raw_val, dict) and "value" in raw_val:
        return raw_val["value"]
    return raw_val


# ── Number coercion ────────────────────────────────────────────────────

def _coerce_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        for sym in ("$", "€", "£", "₹", ","):
            cleaned = cleaned.replace(sym, "")
        cleaned = cleaned.strip()
        if not cleaned:
            return None
        try:
            if "." in cleaned:
                return float(cleaned)
            return int(cleaned)
        except ValueError:
            return None
    return None


# ── Header cleaning ─────────────────────────────────────────────────────

# Standard 15-char Indian GSTIN shape: 2-digit state code, 10-char PAN
# (5 letters + 4 digits + 1 letter), 1 entity code, literal 'Z', 1
# alphanumeric checksum.
_GSTIN_RE = re.compile(r'^\d{2}[A-Z]{5}\d{4}[A-Z]\d Z[A-Z0-9]$'.replace(' ', ''))

# AD (Authorised Dealer) Code: always 14 digits (confirmed real value:
# "6550002-2900009", 7+7). The hyphen is a cosmetic separator, not part of
# the code itself -- some invoices may print it as a plain unhyphenated
# 14-digit run instead, so validity is checked on digit COUNT, not on a
# specific hyphen position (see `_is_valid_ad_code`).
_AD_CODE_SHAPE_RE = re.compile(r'^\d+-?\d+$')


def _repair_gstin(val: str, warnings: list[str] | None = None) -> str:
    """
    A GSTIN is a rigid, checkable 15-character shape, which makes it a
    plausible target for automatic repair (unlike free-text fields, where
    there's no way to tell a wrong value from a right-looking one) -- but
    ONLY for the one specific corruption confirmed on a real invoice: a
    stray extra character glued onto the END of the true GSTIN with no
    gap, from an unrelated adjacent text fragment bleeding onto the same
    line (the same duplicate-text-layer issue that garbles addresses --
    see pdf_reader.py's `_dedupe_overlapping_chars`). Trimming the last
    character and re-checking is safe and unambiguous for that mechanism.

    Deliberately NOT attempted: guessing which position to drop when a
    character got doubled or lost mid-string (confirmed on another real
    file: "AABCS" duplicated to "AABBCS"). The GSTIN shape's character
    classes ([A-Z]{5}, [A-Z0-9] checksum, ...) are loose enough that
    several different single-character removals can each produce a
    structurally valid-looking but DIFFERENT GSTIN -- there's no safe way
    to pick the right one from shape alone, so those are left unrepaired
    and flagged for a human instead of risking a confidently wrong "fix".
    """
    v = str(val).strip().upper().replace(" ", "")
    if not v:
        return v
    if _GSTIN_RE.match(v):
        return v
    if len(v) == 16 and _GSTIN_RE.match(v[:-1]):
        return v[:-1]
    if warnings is not None:
        warnings.append(f"supplier_tax_id '{v}' doesn't match the standard 15-character GSTIN shape")
    return v


def _is_valid_ad_code(v: str) -> bool:
    """14 digits, with or without one hyphen splitting the two halves
    (position not assumed fixed -- only the 7+7 split has been directly
    confirmed, and the hyphen may not appear at all on every invoice)."""
    return bool(_AD_CODE_SHAPE_RE.match(v)) and len(v.replace('-', '')) == 14


def _repair_ad_code(val: str) -> str:
    """
    Confirmed on a real invoice: an AD Code came back with every single
    character doubled consecutively end to end (e.g. "6550002-2900009"
    rendered as "66555500000022--22990000000099") -- the same
    duplicate-text-layer corruption seen elsewhere, just surviving here as
    doubled-but-still-readable text instead of fragmenting into single-char
    tokens. Taking every other character undoes it exactly; only applied
    when doing so produces a value matching the real AD Code shape (see
    `_is_valid_ad_code`), so this can't misfire on a normal, uncorrupted
    value -- hyphenated or not.
    """
    v = str(val).strip()
    if _is_valid_ad_code(v):
        return v
    deduped = v[::2]
    if _is_valid_ad_code(deduped):
        return deduped
    return v


_HEADER_FIELDS = [
    "invoice_number", "invoice_date", "supplier_name", "supplier_address",
    "supplier_tax_id", "buyer_name", "buyer_address", "consignee_name",
    "consignee_address", "currency", "invoice_total", "net_realisable_amount",
    "tax_amount", "tax_rate", "payment_terms", "country_of_origin",
    "country_of_destination", "transit_country", "port_of_loading",
    "port_of_discharge", "export_type_status", "reward_scheme_claim",
    "state_code", "district_code", "ad_code", "iec_number",
    "total_packages", "gross_weight", "net_weight",
]


def _clean_invoice_header(
    header_raw: dict[str, Any], warnings: list[str], shipment_type: str = "export"
) -> dict[str, Any]:
    """
    Unwraps and normalizes invoice-level header fields into a clean flat dict.

    `supplier_tax_id` is only ever a real 15-char Indian GSTIN on an EXPORT
    invoice, where "supplier" is the Indian exporter. On an IMPORT invoice
    "supplier" is the foreign seller, and this same field holds THEIR
    country's own tax-ID format (VAT, EIN, ...) -- there's no fixed shape
    to check it against, so the GSTIN-shape repair/warning and the
    GSTIN-derived state_code cross-check below are both skipped for
    imports rather than misapplying an Indian-only rule to a non-Indian
    value (confirmed real case: a European supplier's VAT number on an
    import invoice was being flagged as a malformed GSTIN).
    """
    clean: dict[str, Any] = {}
    is_export = shipment_type == "export"

    for k in _HEADER_FIELDS:
        val = _unwrap_value(header_raw.get(k, ""))
        if val is None:
            val = ""

        if k == "supplier_tax_id" and val:
            if is_export:
                val = _repair_gstin(val, warnings)
        elif k == "ad_code" and val:
            val = _repair_ad_code(val)
        elif k == "state_code":
            sc_str = str(val).strip() if val else ""
            if sc_str:
                num_match = re.search(r'\((\d{1,2})\)', sc_str)
                if num_match:
                    sc_str = num_match.group(1)
                elif re.match(r'^\d{1,2}$', sc_str):
                    pass
                elif re.search(r'\b(\d{1,2})\b', sc_str):
                    sc_str = re.search(r'\b(\d{1,2})\b', sc_str).group(1)
            val = sc_str
            # Cross-check against the (already-repaired) supplier GSTIN --
            # its first two digits ARE the state code by GSTIN's own spec,
            # a far more reliable source than a free-text "State of
            # Origin" line, which the same duplicate-text-layer corruption
            # can (and did, confirmed on a real file) scatter across
            # several wrong-looking values. Only overrides on a clear
            # mismatch or a missing value -- never touches an
            # already-matching state_code.
            gstin_val = str(clean.get("supplier_tax_id") or "") if is_export else ""
            if len(gstin_val) >= 2 and gstin_val[:2].isdigit():
                gstin_state = gstin_val[:2]
                if not val or val.strip().zfill(2) != gstin_state:
                    if val and val.strip().zfill(2) != gstin_state:
                        warnings.append(
                            f"state_code '{val}' didn't match supplier GSTIN's state "
                            f"prefix '{gstin_state}' -- used the GSTIN's instead"
                        )
                    val = gstin_state
        elif k in ("country_of_origin", "country_of_destination") and val:
            val_str = str(val).strip().upper()
            if val_str in ("INDIA", "IND"):
                val = "IN"
            elif val_str in ("USA", "UNITED STATES", "UNITED STATES OF AMERICA"):
                val = "USA"
        elif k in ("invoice_total", "net_realisable_amount", "tax_amount", "total_packages") and val != "":
            num = _coerce_number(val)
            val = num if num is not None else str(val).strip()
        elif k == "reward_scheme_claim":
            val_str = str(val).strip().upper()
            val = "Y" if any(x in val_str for x in ("Y", "YES", "TRUE", "RODTEP", "MEIS")) else "N"

        clean[k] = val

    return clean


# ── Generic line-item cleaning (template-agnostic) ─────────────────────

_SUBTOTAL_KEYWORDS = {"TOTAL", "SUBTOTAL", "LESS", "NET REALISABLE", "FREE OF COST", "RAW MATERIAL"}


def _clean_generic_item(
    raw_item: dict[str, Any],
    warnings: list[str],
    row_idx: int,
    ser_no: int,
) -> dict[str, Any] | None:
    """
    Cleans one LLM-extracted line item into a generic (not yet
    template-shaped) dict. Returns None if the row should be dropped
    entirely (not a real item -- e.g. a subtotal line).
    """
    if not isinstance(raw_item, dict):
        return None

    item_vals: dict[str, Any] = {}
    for k, v in raw_item.items():
        val = _unwrap_value(v)
        if val is not None:
            item_vals[k] = val

    desc = str(item_vals.get("product_description", "")).upper()
    if any(kw in desc for kw in _SUBTOTAL_KEYWORDS) and "quantity" not in item_vals:
        warnings.append(f"[line_items][row {row_idx+1}] discarded summary/deduction line '{desc}'")
        return None

    qty = _coerce_number(item_vals.get("quantity"))
    price = _coerce_number(item_vals.get("unit_price"))
    total = _coerce_number(item_vals.get("line_total"))
    packages = _coerce_number(item_vals.get("packages"))

    # Mathematical verification: Qty * Price ≈ Line Total
    if isinstance(qty, (int, float)) and isinstance(price, (int, float)) and qty > 0 and price > 0:
        calc_total = round(qty * price, 2)
        if isinstance(total, (int, float)) and total > 0:
            diff = abs(calc_total - total)
            if diff > max(1.0, total * 0.02):
                warnings.append(
                    f"[line_items][row {ser_no}] math discrepancy: "
                    f"Qty({qty}) * Price({price}) = {calc_total} != Line Total({total})"
                )
        elif not total:
            total = calc_total

    # Disambiguate FTA Code vs AD Code vs EU end-use code
    fta = item_vals.get("fta_code")
    end_use = item_vals.get("end_use_code")
    if fta:
        fta_str = str(fta).strip()
        if "AD CODE" in fta_str.upper() or re.match(r'^\d{6,8}[-\s]\d{6,8}$', fta_str):
            fta = None  # AD Code is a bank code, not a tariff/FTA code
        elif fta_str.startswith("GNX") or fta_str.startswith("EU"):
            end_use = end_use or fta_str
            fta = None

    part_no = str(
        item_vals.get("part_number")
        or item_vals.get("part_no")
        or item_vals.get("item_code")
        or item_vals.get("drawing_no")
        or item_vals.get("catalog_no")
        or ""
    ).strip()

    model_type = str(
        item_vals.get("model_or_type")
        or item_vals.get("type")
        or item_vals.get("model")
        or item_vals.get("type_no")
        or item_vals.get("model_no")
        or ""
    ).strip()

    raw_desc = str(item_vals.get("product_description", "")).strip()
    if len(model_type) > 40:
        model_type = ""
    elif model_type.lower() == raw_desc.lower() or (len(raw_desc) > 8 and model_type.lower() in raw_desc.lower()):
        model_type = ""

    ohms_val = item_vals.get("ohms") or item_vals.get("rating") or item_vals.get("spec")
    if ohms_val and str(ohms_val) not in str(model_type):
        model_type = f"{model_type} ({ohms_val} OHMS)" if model_type else f"{ohms_val} OHMS"

    extracted_sno = item_vals.get("item_ser_no")
    actual_sno = extracted_sno if extracted_sno not in (None, "") else ser_no

    return {
        "item_ser_no": actual_sno,
        "product_description": raw_desc,
        "part_number": part_no,
        "model_or_type": model_type,
        "order_number": str(item_vals.get("order_number") or "").strip(),
        "hsn_code": item_vals.get("hsn_code", ""),
        "quantity": qty if qty is not None else "",
        "unit_of_measurement": item_vals.get("unit_of_measurement", ""),
        "unit_price": price if price is not None else "",
        "line_total": total if total is not None else "",
        "end_use_code": end_use or "",
        "fta_code": fta or "",
        "country_of_origin": item_vals.get("country_of_origin", ""),
        "igst_amount": item_vals.get("igst_amount", ""),
        "igst_rate": item_vals.get("igst_rate", ""),
        "packages": packages if packages is not None else "",
    }


# ── Type-specific mappers: the ONLY two places that know a template's ──
# ── column names. Each returns just the columns it can populate; every ──
# ── other column the template defines comes back "" via ──
# ── validate_row_against_model(), never a guessed default. ─────────────

def _map_export_item(item: dict[str, Any], header: dict[str, Any]) -> dict[str, Any]:
    """Cleaned generic item + cleaned header -> export template's ITEM row."""
    def _hdr(k: str) -> Any:
        return header.get(k) or ""

    return {
        "Item_Ser_No": item["item_ser_no"],
        "Item_RITC": item["hsn_code"],
        "Item_Desc": item["product_description"],
        "Item_Unit1": item["unit_of_measurement"],
        "Item_Qty": item["quantity"],
        "Item_Unit_Price": item["unit_price"],
        "Item_Reward": _hdr("reward_scheme_claim"),
        "Itm_Source_Cntry": item["country_of_origin"] or _hdr("country_of_origin"),
        "Itm_Transit_Cntry": _hdr("transit_country"),
        # This item's OWN package count when the invoice prints one per
        # item (confirmed real case: two items, 108 and 504 packages on
        # different pallet ranges -- each must keep its own number, not
        # the header's whole-shipment total copied onto both). Falls back
        # to the header total only when this item never got its own count
        # extracted, e.g. invoices that print just one whole-shipment
        # figure with no per-item breakdown -- same behavior as before
        # this field existed.
        "ItmTotpkg": item["packages"] if item["packages"] != "" else _hdr("total_packages"),
        "ItmIGSTstatus": _hdr("export_type_status"),
        "ItmTaxableVal": item["line_total"],
        "ItmIGSTamt": item["igst_amount"],
        "Item_End_Use": item["end_use_code"],
        "ItmIGSTPer": item["igst_rate"] or _hdr("tax_rate"),
        "State Code": _hdr("state_code"),
        "District Code": _hdr("district_code"),
        "FTA Code": item["fta_code"],
        "Qty Tariff": item["quantity"],
        "Unit Tariff": item["unit_of_measurement"],
    }


def _map_import_item(item: dict[str, Any], header: dict[str, Any]) -> dict[str, Any]:
    """
    Cleaned generic item + cleaned header -> import template's BOE row.

    Only the commercially-derivable subset of BOE's ~85 columns is
    populated here -- the rest (BCD/CVD/SWC/IGST notification numbers,
    SIMS registration, ADIC references, ...) are customs-filing-stage
    data, not something printed on a commercial invoice. Same honest
    "usually empty" situation as the export template's DRAWBACK/License/
    STR/etc. sheets.
    """
    def _hdr(k: str) -> Any:
        return header.get(k) or ""

    country = item["country_of_origin"] or _hdr("country_of_origin")

    return {
        "SL_No": item["item_ser_no"],
        "Item_RITC": item["hsn_code"],
        "Item_Desc1": item["product_description"],
        "Item_End_Use": item["end_use_code"],
        "Item_Country_Org": country,
        "Item_Qty": item["quantity"],
        "Item_Unit": item["unit_of_measurement"],
        "Item_Unit_Price": item["unit_price"],
        "Item_Taxable_Val": item["line_total"],
        "Qty_Tariff": item["quantity"],
        "Unit_Tariff": item["unit_of_measurement"],
        "Model": item["model_or_type"],
        "COO Country": country,
        "Transit Country": _hdr("transit_country"),
    }


_MAPPERS = {"export": _map_export_item, "import": _map_import_item}


# ── Domain-specific post-mapping cleanup ────────────────────────────────

# ISO container code: 4 letters (owner + category) + 7 digits
_CONTAINER_PATTERN = re.compile(r'^[A-Z]{3}[UJZ]\d{6,7}$')

# HSN/tariff description appended to item description, e.g.
# "870899B OTHERS" or "8483B TRANSMISSION SHAFTS..."
_HSN_DESC_TAIL = re.compile(r'\s+\d{4,8}[A-Z]\s+[A-Z].*$', re.IGNORECASE)

_NA_LIKE = ("NA", "N/A", "-", "NIL", "NONE", "")


def _trim_hsn_desc_tail(row: dict[str, Any], desc_field: str, sheet_name: str, warnings: list[str], row_idx: int) -> None:
    desc = row.get(desc_field)
    if desc is None:
        return
    desc_str = str(desc)
    match = _HSN_DESC_TAIL.search(desc_str)
    if match:
        clean = desc_str[:match.start()].strip()
        if clean:
            row[desc_field] = clean
            warnings.append(
                f"[{sheet_name}][row {row_idx}] trimmed HSN tariff description from {desc_field}"
            )


def _apply_export_domain_rules(row: dict[str, Any], warnings: list[str], row_idx: int) -> None:
    """India-customs cleanup specific to the export ITEM sheet's columns."""
    sheet_name = "ITEM"

    # ItmHawb should be an airway bill, not a container number
    hawb = row.get("ItmHawb")
    if hawb is not None:
        hawb_str = str(hawb).strip().upper()
        if _CONTAINER_PATTERN.match(hawb_str):
            warnings.append(
                f"[{sheet_name}][row {row_idx}] removed container number "
                f"'{hawb_str}' from ItmHawb (not an airway bill)"
            )
            row["ItmHawb"] = ""
        elif hawb_str in _NA_LIKE:
            row["ItmHawb"] = ""

    _trim_hsn_desc_tail(row, "Item_Desc", sheet_name, warnings, row_idx)

    # FTA Code vs AD Code vs EU end-use code
    fta = row.get("FTA Code")
    if fta:
        fta_str = str(fta).strip()
        if fta_str.startswith("GNX") or fta_str.startswith("EU"):
            if not row.get("Item_End_Use"):
                row["Item_End_Use"] = fta_str
                warnings.append(
                    f"[{sheet_name}][row {row_idx}] moved EU Code '{fta_str}' from FTA Code to Item_End_Use"
                )
            row["FTA Code"] = ""
        elif "AD CODE" in fta_str.upper() or re.match(r'^\d{6,8}[-\s]\d{6,8}$', fta_str) or re.match(r'^\d{7,}$', fta_str):
            warnings.append(
                f"[{sheet_name}][row {row_idx}] removed AD Code '{fta_str}' from FTA Code "
                f"(AD Code is a bank code, not an FTA Code)"
            )
            row["FTA Code"] = ""

    for field in ("Itm_Transit_Cntry", "Itm_Source_Cntry"):
        val = row.get(field)
        if val is not None and str(val).strip().upper() in _NA_LIKE:
            row[field] = ""

    # Normalize Y/N flag fields (only if actually populated -- the mapper
    # doesn't set these itself, so this only fires if something upstream did)
    for field in ("item_Cess", "item_accessory", "item_thirdparty", "item_Quota",
                  "item_AR4", "Item_Reward", "Item_STR", "Item_JNoti_No"):
        if row.get(field):
            val_str = str(row[field]).strip().upper()
            if field == "Item_Reward" and any(k in val_str for k in ("RODTEP", "MEIS", "REWARD", "CLAIM", "SCHEME")):
                row[field] = "Y"
            elif val_str in ("TRUE", "YES", "1", "Y"):
                row[field] = "Y"
            elif val_str in ("FALSE", "NO", "0", "0.0", "N", *_NA_LIKE):
                row[field] = ""
            else:
                row[field] = "Y" if val_str.startswith("Y") else ""

    for cess_field in ("CESS Per", "Cess Rate", "Cess Amount"):
        if row.get(cess_field) in (0, 0.0, "0", "0.0"):
            row[cess_field] = ""

    state_code = row.get("State Code")
    if state_code:
        sc_str = str(state_code).strip()
        num_match = re.search(r'\((\d{1,2})\)', sc_str)
        if num_match:
            row["State Code"] = num_match.group(1)
        elif re.search(r'\b(\d{1,2})\b', sc_str):
            row["State Code"] = re.search(r'\b(\d{1,2})\b', sc_str).group(1)


def _apply_import_domain_rules(row: dict[str, Any], warnings: list[str], row_idx: int) -> None:
    """India-customs cleanup specific to the import BOE sheet's columns."""
    sheet_name = "BOE"

    _trim_hsn_desc_tail(row, "Item_Desc1", sheet_name, warnings, row_idx)

    for field in ("Item_Country_Org", "COO Country", "Transit Country"):
        val = row.get(field)
        if val is not None and str(val).strip().upper() in _NA_LIKE:
            row[field] = ""


_DOMAIN_RULES = {"export": _apply_export_domain_rules, "import": _apply_import_domain_rules}


def _drop_packing_list_duplicates(
    items: list[dict[str, Any]], warnings: list[str]
) -> list[dict[str, Any]]:
    """
    Deterministic backstop for a real failure mode (llm_extract.py's prompt
    now also warns against this directly, but a prompt rule alone isn't a
    guarantee): some invoices bundle a packing list alongside the
    commercial invoice, repeating the same part number/quantity under a
    Net Wt./Gross Wt./box-breakdown table instead of a price table. If the
    model still emits that repeat as its own line item, it comes back with
    the same part_number and quantity as a real item but no usable price
    (confirmed on a real file: two rows for the same part, same qty=336,
    one with unit_price=560.0, the other with unit_price/line_total both
    empty). Drop the priceless duplicate rather than keep a phantom row.
    """
    priced_keys = {
        (it.get("part_number"), it.get("quantity"))
        for it in items
        if it.get("part_number") and it.get("unit_price") not in (None, "")
    }
    kept: list[dict[str, Any]] = []
    for it in items:
        key = (it.get("part_number"), it.get("quantity"))
        if key in priced_keys and it.get("unit_price") in (None, ""):
            warnings.append(
                f"[line_items] dropped priceless duplicate of part '{it.get('part_number')}' "
                f"(qty {it.get('quantity')}) -- looks like a packing-list repeat, not a new item"
            )
            continue
        kept.append(it)
    return kept


def _renumber_items_if_ser_no_resets(items: list[dict[str, Any]], warnings: list[str]) -> None:
    """
    Deterministic backstop for the same failure mode llm_extract.py's
    prompt now warns against directly (item_ser_no rule 3): on an invoice
    with NO real serial-number column at all, confirmed to have items
    grouped under repeating section headers (e.g. "PO No 240300", "PO No
    240400") with no SNo column anywhere, the model is supposed to leave
    item_ser_no null on every row -- validate_and_coerce()'s own counter
    then fills in a clean, never-resetting 1..N sequence. But a prompt
    rule alone isn't a guarantee across a long multi-page table: on a
    real file, one page's worth of items (out of many pages that all
    correctly came back null) still got a hallucinated LOCAL count that
    restarted at 1, landing in the middle of the merged list as
    ...,79,80,1,2,3,...,22,92,... instead of a continuing sequence.

    A later item_ser_no that is <= an earlier one in the FINAL merged
    list is exactly that signal -- a genuinely-printed, single serial
    column never resets or repeats across a document's own item table.
    When that's detected anywhere, none of the model's item_ser_no
    values for this document can be trusted as a real printed sequence,
    so this discards ALL of them (not just the offending ones -- a
    partially-trusted mix would just move the discontinuity) and
    replaces every item's item_ser_no with a clean 1..N count in final
    list order. Mutates `items` in place; a no-op when the sequence is
    already consistent (the overwhelming majority of invoices).
    """
    prev = None
    resets = False
    for it in items:
        v = it.get("item_ser_no")
        if v in (None, ""):
            continue
        try:
            v_num = float(v)
        except (TypeError, ValueError):
            continue
        if prev is not None and v_num <= prev:
            resets = True
            break
        prev = v_num

    if not resets:
        return

    warnings.append(
        "[line_items] item_ser_no reset or repeated partway through the merged "
        "list -- renumbered every item 1..N instead of trusting a partial sequence"
    )
    for i, it in enumerate(items, start=1):
        it["item_ser_no"] = i


# ── Main entry point ────────────────────────────────────────────────────

def validate_and_coerce(
    raw: dict[str, Any],
    schema: dict[str, dict[str, Any]],
    shipment_type: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[str], dict[str, Any]]:
    """
    Turns the generic {"invoice_header": ..., "line_items": ...} shape
    produced by llm_extract's extract_invoice_header()/extract_line_items()
    (assembled by reextraction.py's retry loop) into template-shaped
    sheets for the given `shipment_type` ("export" or "import").

    Returns (sheets, warnings, header) where `sheets` has one key per sheet
    in the loaded template ("ITEM" or "BOE" holds the mapped line items;
    every other sheet is [] -- customs-filing data this pipeline never
    populates, same as always).
    """
    if shipment_type not in _MAPPERS:
        raise ValueError(f"shipment_type must be 'export' or 'import', got {shipment_type!r}")

    warnings: list[str] = []
    raw_header = raw.get("invoice_header", {}) if isinstance(raw, dict) else {}
    raw_items = raw.get("line_items", []) if isinstance(raw, dict) else []

    cleaned_header = _clean_invoice_header(raw_header, warnings, shipment_type)

    generic_items: list[dict[str, Any]] = []
    ser_no = 1
    for idx, raw_item in enumerate(raw_items):
        cleaned = _clean_generic_item(raw_item, warnings, idx, ser_no)
        if cleaned is None:
            continue
        generic_items.append(cleaned)
        ser_no += 1

    generic_items = _drop_packing_list_duplicates(generic_items, warnings)
    _renumber_items_if_ser_no_resets(generic_items, warnings)

    item_sheet_name = ITEM_SHEET_NAME[shipment_type]
    mapper = _MAPPERS[shipment_type]
    domain_rules = _DOMAIN_RULES[shipment_type]

    sheets: dict[str, list[dict[str, Any]]] = {name: [] for name in schema}

    if item_sheet_name not in schema:
        warnings.append(
            f"loaded template has no '{item_sheet_name}' sheet -- line items could not be placed"
        )
    else:
        sheet_models = build_sheet_models(schema)
        item_model = sheet_models.get(item_sheet_name)
        item_fields = schema[item_sheet_name]["fields"]

        mapped_rows: list[dict[str, Any]] = []
        for i, generic in enumerate(generic_items):
            mapped = mapper(generic, cleaned_header)
            domain_rules(mapped, warnings, i)

            if item_model:
                validated = validate_row_against_model(
                    mapped, item_model, item_sheet_name, i, warnings, all_fields=item_fields
                )
            else:
                validated = mapped

            if validated is not None:
                validated["part_number"] = generic["part_number"]
                validated["model_or_type"] = generic["model_or_type"]
                validated["order_number"] = generic["order_number"]
                mapped_rows.append(validated)

        sheets[item_sheet_name] = mapped_rows

    return sheets, warnings, cleaned_header
