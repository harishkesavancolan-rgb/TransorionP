"""
llm_extract.py
---------------
Calls the LLM to turn invoice text into two generic, template-agnostic
JSON payloads (header metadata + line items) and reports token cost.

This module knows nothing about customs template column names -- mapping
these generic fields onto the export ITEM sheet or the import BOE sheet is
validate.py's job (see `_map_export_item` / `_map_import_item` there),
since which template applies depends on invoice type, which is decided
before this module is even called (see detect_type.py).

Key properties:
- Two passes (header, line items) run CONCURRENTLY via asyncio.gather.
- Network calls are retried with backoff on transient errors (rate limits,
  connection drops, timeouts, 5xx) -- NOT on auth/bad-request errors.
- `_extract_json` does multi-stage JSON repair since LLMs occasionally
  emit near-JSON (bare N/A, trailing commas, markdown fences, ...).
"""
from __future__ import annotations
import json
import logging
import os
import re
from typing import Any
from dotenv import load_dotenv
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger("invoice_extractor")

# Reasoning models (o1, o3, o3-mini, o4-mini, gpt-5*, ...) don't support
# response_format={"type": "json_object"} -- they need plain-text JSON
# instructions instead. Everything else (gpt-4.1-*, gpt-4o*, ...) supports
# standard JSON mode and gets it, since it's a stronger guarantee.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

# Retries only transient, retry-worthy failures. Auth errors, bad requests,
# and content-policy rejections are never retried -- they won't succeed on
# a second attempt and would just waste the backoff window.
_llm_retry = retry(
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    ),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(m.startswith(p) for p in _REASONING_MODEL_PREFIXES)


# ── Token Cost Pricing (USD per 1 Million tokens) ──
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5-nano": {
        "input":        0.05,
        "cached_input": 0.005,
        "output":       0.40,
    },
    "gpt-4.1-nano": {
        "input":        0.10,
        "cached_input": 0.025,
        "output":       0.40,
    },
}

# USD → INR conversion rate (update as needed)
_USD_TO_INR = 96.34


def _calculate_cost(
    model: str,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    """
    Calculates token cost in both USD and INR.
    Returns a dict with token counts and cost breakdown.
    """
    pricing = _MODEL_PRICING.get(model)
    if not pricing:
        # Fallback: use gpt-4.1-nano pricing for unknown models
        pricing = _MODEL_PRICING["gpt-4.1-nano"]

    # Non-cached input tokens = total input - cached portion
    non_cached_input = max(0, input_tokens - cached_tokens)

    cost_input_usd = (non_cached_input / 1_000_000) * pricing["input"]
    cost_cached_usd = (cached_tokens / 1_000_000) * pricing["cached_input"]
    cost_output_usd = (output_tokens / 1_000_000) * pricing["output"]
    total_cost_usd = cost_input_usd + cost_cached_usd + cost_output_usd
    total_cost_inr = total_cost_usd * _USD_TO_INR

    return {
        "input_tokens": non_cached_input,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(total_cost_usd, 6),
        "cost_inr": round(total_cost_inr, 4),
    }


def _extract_usage(resp) -> dict[str, int]:
    """Extracts token usage from an OpenAI API response."""
    usage = getattr(resp, "usage", None)
    if not usage:
        return {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}

    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    # Extract cached tokens from prompt_tokens_details if available
    cached_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0

    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
    }


def _extract_json(raw: str) -> dict[str, Any] | list[Any]:
    """
    Parses the model's JSON response with multi-stage sanitization and repair:
    1. Direct JSON parse
    2. json_repair.loads (handles unquoted N/A, bare words, trailing commas, missing braces, etc.)
    3. Stripping markdown fences and extracting outermost {...} or [...]
    4. Regex sanitization for unquoted N/A, NA, None, undefined, NaN, trailing commas
    5. Python literal evaluation fallback (ast.literal_eval)
    """
    if not raw or not raw.strip():
        return {}

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text).strip()

    # 1. Direct standard JSON parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. json_repair (industry standard bulletproof repair for LLM outputs)
    try:
        import json_repair
        repaired = json_repair.loads(text)
        if isinstance(repaired, (dict, list)) and repaired:
            return repaired
    except Exception:
        pass

    # 3. Extract outermost { ... } or [ ... ]
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    start_arr = text.find("[")
    end_arr = text.rfind("]")

    candidate = text
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        if start_arr == -1 or start_obj < start_arr:
            candidate = text[start_obj:end_obj + 1]
    elif start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = text[start_arr:end_arr + 1]

    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Try json_repair on candidate
    try:
        import json_repair
        repaired = json_repair.loads(candidate)
        if isinstance(repaired, (dict, list)) and repaired:
            return repaired
    except Exception:
        pass

    # 4. Repair common LLM JSON syntax errors:
    # a. Bare unquoted N/A, NA, None, undefined, NaN (e.g. "value": N/A -> "value": null)
    sanitized = re.sub(r':\s*(?:N/A|n/a|NA|na|None|undefined|NaN|-|\?)\b', ': null', candidate)
    # b. Empty value before comma or brace (e.g. "value": , -> "value": null,)
    sanitized = re.sub(r':\s*,\s*', ': null, ', sanitized)
    # c. Trailing commas before closing braces or brackets (e.g. { "a": 1, } -> { "a": 1 })
    sanitized = re.sub(r',\s*([}\]])', r'\1', sanitized)
    # d. Python booleans
    sanitized = re.sub(r'\bTrue\b', 'true', sanitized)
    sanitized = re.sub(r'\bFalse\b', 'false', sanitized)

    try:
        return json.loads(sanitized)
    except Exception:
        pass

    # 5. Fallback to ast.literal_eval if LLM used Python dictionary / list syntax
    import ast
    try:
        pythonic = (
            candidate.replace("null", "None")
            .replace("true", "True")
            .replace("false", "False")
        )
        eval_res = ast.literal_eval(pythonic)
        if isinstance(eval_res, (dict, list)):
            return eval_res
    except Exception:
        pass

    # 6. Final attempt on sanitized string or raise with clear context
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError as err:
        logger.error(f"Failed to parse LLM JSON output. Raw snippet: {raw[:300]!r}")
        raise err


_HEADER_SYSTEM_PROMPT = """\
You are an expert commercial invoice understanding AI.
You receive the complete, layout-preserved 2D spatial text extracted from an invoice / export document.
The text strictly preserves vertical alignment, multi-column separation, horizontal gutters, and word linearity without word collisions.
A "│" character marks a detected column boundary -- text on either side of it belongs to separate fields/columns on that line. Never merge text across a "│" into one value or one sentence.

Extract document-level header and summary metadata into a strict, FLAT JSON object.
Do NOT wrap values in {"value": ..., "confidence": ...}. Return primitive JSON values directly (strings, numbers, null).

Fields to extract:
- invoice_number: Invoice / Bill number ONLY (string or null). Some invoices print the number and date together under one shared column/label like "Invoice No. & Date" (e.g. "KA/2627/I/016909/15-JUN-26"). If so, that trailing date-like token is NOT part of the invoice number -- strip it here and put it in invoice_date instead, so invoice_number for that example is just "KA/2627/I/016909". If the document genuinely states MORE THAN ONE invoice number (e.g. a single combined document covering two paired invoice numbers, such as "KA/2627/I/016909" and "KA/2627/I/016910" both printed in the Invoice No. field), include ALL of them here (each with its own date suffix already stripped) joined by " / " in the order printed -- do NOT pick only one and silently drop the rest. BUT the SAME number printed again on other pages (bank-details sheet, packing list, transporter copy, page header) is ONE invoice -- list it once, even if OCR spelled it slightly differently ("SI3626204603" vs "S13626204603"). Join numbers with " / " only for genuinely DIFFERENT invoices, each with its own items and total.
- invoice_date: Invoice date as printed or in YYYY-MM-DD format (string or null). If multiple invoice numbers were combined into invoice_number above, each with its own date, join their dates here the same way and in the same order, so date N corresponds to invoice number N.
- supplier_name: Exporter / Seller legal company name (string or null)
- supplier_address: Exporter complete multi-line address verbatim (from plot/building number, street, industrial area, city, state, postal PIN code, and country). Do NOT omit address lines or postal codes. (string or null)
- supplier_tax_id: Supplier's tax identification number -- a real Indian GSTIN on an EXPORT invoice (e.g. "27AAACG8030H2ZR"), or the foreign supplier's own tax ID (VAT, EIN, etc.) on an IMPORT invoice (string or null)

*** BUYER vs CONSIGNEE SEPARATION (CRITICAL) ***:
- buyer_name: Buyer / Importer legal company name (under "Buyer", "Buyer (if other than Consignee)", "Bill To", "Sold To") (string or null)
- buyer_address: Address of Buyer ONLY. Capture the entire multi-line address verbatim from building/street to zip/postal code and country. Do NOT truncate. (string or null)
- consignee_name: Consignee / Delivery party legal name (under "Consignee", "Ship To", "Deliver To") (string or null)
- consignee_address: Address of Consignee ONLY. Capture the entire multi-line address verbatim. Do NOT truncate. (string or null)

Notice: Side-by-side columns (e.g. Consignee on the left, Buyer in the middle, Shipping Marks on the right) are cleanly separated by horizontal whitespace. Read each column block independently and DO NOT merge their addresses. Keep reading YOUR column all the way down through every line the address spans, even on a line where the OTHER column's content is clearly unrelated (e.g. a different party's own country name) -- do not stop early just because that line "looks like" it belongs elsewhere. Confirmed real miss: a buyer address's own country line ("United States") sat on the same text row as the ship-to column's country ("France"), separated by a "│" column marker -- every other line of that buyer address was captured correctly, but that one line was skipped, silently dropping the country from the address. Read the full column height, line by line, not just the lines whose other-column content also happens to be blank or related.
IMPORTANT: consignee_name and buyer_name are two DIFFERENT company names printed in two DIFFERENT horizontal positions on the SAME line (e.g. "Consignee: NORTH AMERICAN PRODUCTION" on the left, "Buyer (if other than Consignee)" as a separate label further right, with the actual buyer name appearing on the NEXT line under that right-hand label). NEVER concatenate the buyer's name onto the end of consignee_name (or vice versa) just because they appear on the same or adjacent lines -- each belongs only under its own label, stopping at the start of the next column's horizontal position.

IMPORTANT: supplier_address / buyer_address / consignee_address must contain ONLY physical address lines (building/street/area/city/state/postal code/country) -- NEVER a Tax ID, GSTIN, CIN, Registration Number, or similar identifier, even when it's printed glued to the same line as an address line, or stacked directly below the address in the same column with no visible label separating them. Confirmed real misses: a buyer's own postal code had "TAX ID : 83-2387668" printed right after it on the same line with no separator ("53188-1615  TAX ID : 83-2387668") and that tax ID got wrongly appended into buyer_address; a supplier's address block had "C Ex Reg No:", "GSTIN :", and "CIN:" lines stacked immediately below it in the same left column, and those got wrongly absorbed into supplier_address. Recognize these by their own label/pattern (a "TAX ID"/"GSTIN"/"CIN"/"Reg No" prefix, or a code matching a GSTIN/CIN/registration-number shape) and stop the address there -- route GSTIN/tax-ID specifically to supplier_tax_id if that's what it is, and simply omit the rest (CIN, Reg No) if there is no dedicated field for it, rather than letting it bleed into the address string.

- currency: Invoice Currency code (e.g. "USD", "EUR", "INR", "GBP") (string or null). Take it from the code printed next to the INVOICE's own amounts (e.g. "INR 2,527.70" in the unit price / total columns). The "Currency" box on an air waybill or bill of lading is the transport-charge currency, NOT the invoice currency -- do not use it when the invoice prints its own.
- invoice_total: The actual amount chargeable/payable to the buyer, in the SAME currency you put in `currency` above (numeric or null). ONLY extract a value that is EXPLICITLY PRINTED on the document itself under a label such as "Amount Chargeable", "Grand Total", "Total Value" or "Total Invoice Value". NEVER compute, sum, derive, or otherwise deduce this field from the line items yourself, even as a fallback when no explicit total is printed -- if there is no explicitly labeled total anywhere on the document, leave invoice_total null. A number you calculated is not the same thing as a number the document states, and this field must only ever hold the latter; a separate, deterministic check elsewhere reconciles the printed total against the line items -- that is not your job, and guessing at it yourself only hides a real mismatch instead of surfacing it. CRITICAL: NEVER use a "Total F.O.B Value" / "FOB Value" line for this field. On Indian export invoices that value is routinely printed in INR for customs/RBI declaration purposes even when the invoice's trade currency is USD/EUR/etc -- using it here silently mixes a rupee figure into a field tagged with the wrong currency. If the only total-like number you can find is explicitly labeled "F.O.B Value", leave invoice_total null rather than use it. CRITICAL -- if invoice_number above holds MORE THAN ONE invoice number (a combined document): do NOT sum the invoices' totals into one number. Confirmed real case: two FULLY SEPARATE, independently-numbered invoices (each its own header, its own items, its own explicitly printed "TOTAL INVOICE AMOUNT" line -- 4,416.00 for one, 1,472.00 for the other) were bundled into a single PDF; the correct invoice_total is "4416.00 / 1472.00", NOT "5888" (their sum) -- 5888 is not a real amount printed anywhere and doesn't belong to either invoice. Instead, find EACH sub-invoice's own explicitly printed total (same labels as above: "Amount Chargeable", "Grand Total", "Total Invoice Value", "Total Invoice Amount") and join them with " / " in the SAME order as their invoice numbers in invoice_number, so total N corresponds to invoice number N -- exactly like invoice_date does. If a given sub-invoice has no total of its own explicitly printed, its own position in this " / "-joined value is null, not a computed figure.
- net_realisable_amount: Net Realisable / Net Chargeable Amount after deductions (numeric or null). CRITICAL: Extract ONLY if explicitly labeled as "Net Realisable", "Net Chargeable", or "Realizable Value". NEVER map "Assessable Value", "Taxable Value", or "Total Amount" into this field. If not explicitly labeled as net realisable/chargeable, return null.
- tax_amount: Total Tax / IGST Amount (numeric or null)
- tax_rate: Tax Rate (e.g. "18%") if explicitly stated (string or null)
- payment_terms: Terms of Payment ONLY -- credit/payment condition (e.g. "Net 60 days", "45 DAYS", "IMMEDIATE", "IC60"). There is no separate field for Incoterms/delivery terms, but that does NOT mean they belong here -- NEVER put an Incoterm (e.g. "FCA", "FOB", "CIF", "DAP", "EXW", "EX WORKS-Factory") into payment_terms, even when the invoice prints it right next to or wrapped onto the same value as the real payment term under one shared label like "Terms of Delivery & Payment:". Some invoices print "Incoterms" and "Payment Terms" as two SEPARATE, side-by-side labeled columns (e.g. "Incoterms: DAP" next to "Payment Terms: IC60") -- take only the Payment Terms column there. Others combine both under one label with the value wrapping onto a following line (e.g. "IMMEDIATE" then "EX WORKS-Factory (INCO terms 2000)" on the next line) -- in that case too, take ONLY the payment-term fragment ("IMMEDIATE") for payment_terms and OMIT the Incoterm fragment entirely; there is no field to put it in, so drop it rather than let it dilute payment_terms with a different concept. (string or null)
- country_of_origin: Country of origin of goods (e.g. "INDIA", "IN", "36") (string or null)
- country_of_destination: Country of final destination (e.g. "USA", "GERMANY") (string or null)
- transit_country: Intermediate transit country if specified (string or null)
- port_of_loading: Port of loading / airport (e.g. "NHAVA SHEVA, INDIA", "MUMBAI") (string or null)
- port_of_discharge: Port of discharge / airport (e.g. "HOUSTON, USA", "SANDIEGO") (string or null)
- export_type_status: GST export status: "LUT" (for Bond/LUT exports), "PAID" (if IGST paid), or "NA" (string or null)
- reward_scheme_claim: "Y" if invoice contains RoDTEP / MEIS reward claim statement, otherwise "N"
- state_code: Exporter numeric state code (e.g. "12" from "MH(12)", or "27" from GSTIN "27...") (string or null)
- district_code: District code if printed (string or null)
- ad_code: Authorised Dealer Code (e.g. "0301216-6000009") (string or null)
- iec_number: Importer Exporter Code (IEC) (string or null)
- total_packages: Total number of physical packages -- the innermost physical containers the goods are packed into (cartons, cases, boxes, packages, bags, drums, crates -- whichever word THIS invoice actually uses), a SEPARATE, usually much SMALLER count than the piece/unit quantity, NOT the same number. On these invoices it prints in the item table's own "TOTAL" row, under a "No. of Pkgs" column that sits BEFORE the "Quantity (PCS)" column -- e.g. a row reading "TOTAL   1        336" has 336 as the piece quantity and 1 (the number appearing FIRST, right after "TOTAL") as total_packages; do not take the second, larger number for this field just because it is also on the TOTAL row. Confirmed real misses: this field came back holding the piece quantity itself (e.g. 336 or 1680) when the true package count printed on the same row was a single-digit or low double-digit number (1, 6, 11). If the "No. of Pkgs" column position on the TOTAL row is blank -- nothing printed between "TOTAL" and the quantity figure -- leave this field null; do NOT substitute the quantity or invent a count. Other invoices instead describe packing in a sentence rather than a table row, e.g. "Four Nos Of Pallet Containing 3840 Nos (20 Nos Each in 192 carton boxes)" -- that sentence names THREE different counts (3840 pieces = the quantity, 4 pallets, 192 carton boxes), and total_packages is the carton/case/box/package count (192 here), never the pallet count (a larger consolidation/handling unit, not what this field means) and never the piece quantity. (numeric or null)
- gross_weight: Total gross weight with unit (e.g. "6,375.24 KG", "12.00 Kgs") (string or null)
- net_weight: Total net weight with unit (e.g. "6,240.24 KG", "10.00 Kgs") (string or null)

RULES:
1. Return a FLAT JSON object mapping field names directly to primitive values (strings, numbers, or null).
2. DO NOT wrap fields in {"value": ..., "confidence": ...}.
3. If a field is not present in the document, set value to null or "".
4. NEVER write bare unquoted words like N/A, NA, or None (which are invalid JSON). Use JSON null or "" instead.
5. NEVER guess or calculate unmentioned values.
"""

_LINE_ITEMS_SYSTEM_PROMPT = """\
You are an expert commercial invoice table extractor.
You receive the complete, layout-preserved 2D spatial text extracted from an invoice.
The table rows and columns strictly maintain vertical alignment, linear gutters, and horizontal spacing without word collisions.
A "│" character marks a detected column boundary -- text on either side of it is a separate table cell/column. Never merge text across a "│" into one field (e.g. never combine a description with the next column's code/qty/price just because they're on the same row).

Extract EVERY distinct line item from the invoice table into a JSON object containing a FLAT array of line item objects under the key "line_items".

Return format:
{
  "line_items": [
    {
      "item_ser_no": 1,
      "product_description": "ROTOR SHAFT ASY",
      "part_number": "1004128",
      "model_or_type": "F2",
      "order_number": "PO-84920",
      "hsn_code": "84831099",
      "quantity": 100,
      "unit_of_measurement": "NOS",
      "unit_price": 50.0,
      "line_total": 5000.0,
      "end_use_code": null,
      "fta_code": null,
      "packages": null,
      "country_of_origin": null
    }
  ]
}

CRITICAL RULES FOR LINE ITEMS:
1. Extract EVERY distinct item row individually across all pages. Do NOT miss any rows and NEVER collapse multiple items into one!
2. Return flat primitive values directly (strings, numbers, null). DO NOT wrap values in {"value": ..., "confidence": ...}.
3. item_ser_no: Extract the EXACT Item Serial / Line Number as printed in the invoice table (e.g. 1, 2, 3... or 10, 20, 30... or 0010, 0020...). If no SNo column is printed on the invoice, set to null. Confirmed real case: an invoice's items were grouped under repeating section headers like "PO No 240300" / "PO No 240400", each followed by a handful of item rows with columns Part No / Description / Qty / Uom / Unit / Amount -- there was no serial-number column anywhere. A "PO No" (or similar) heading that groups rows underneath it is NOT an SNo column, and the row's position under that heading is NOT a serial number either -- do NOT invent your own row-count that increments within each group and resets to 1 at the next heading. In that case item_ser_no must be null on every single row; the caller assigns a real, non-resetting sequence number afterward.
4. Do NOT include subtotal, deduction, or total rows (e.g. "TOTAL", "SUBTOTAL", "LESS: RAW MATERIALS", "NET REALISABLE") as line items. Likewise a FREIGHT / SHIPPING / INSURANCE / HANDLING / PACKING charge printed once below the table as "FREIGHT: <amount>" (one label, one amount, no quantity or unit price) is a charge, NOT a line item -- skip it.
5. product_description: The commercial item name (e.g. "ROTOR SHAFT ASY", "TO-5 COILS", "ELECTRONIC COMPONENTS"). Omit generic tariff chapter descriptions. The description column frequently WRAPS across multiple physical rows -- both ABOVE and BELOW the specific row that carries the item's own quantity/rate/amount (per rule 17). Read the description column's text on every row belonging to this item (the row with the quantity/rate/amount, plus any row directly above or below it that has no quantity/rate/amount of its own but has description-column text) and CONCATENATE all of it in reading order into one complete description -- do not take only the fragment that happens to share a row with the quantity/rate/amount. Example: "Discovery" on the row above, "IQ 2Ring-Gen2 India" on the row with the quantity/rate/amount -> full description is "Discovery IQ 2Ring-Gen2 India". Example: "AXIAL HEADHOLDER FOR HSA AND NP" on rows above the quantity/rate/amount row, "TABLES-RoHS" on the row below it -> full description is "AXIAL HEADHOLDER FOR HSA AND NP TABLES-RoHS". IMPORTANT -- this vertical (above/below) concatenation is NEVER a license to concatenate HORIZONTALLY across a "│" column boundary: some invoices print the description TWICE side by side as two SEPARATE, fully-labeled columns in different languages (e.g. "Description Produit" in French immediately followed by "Description of Goods" in English, each its own column, separated by "│"). That is not a wrapped/continued value -- it is the SAME field printed twice for two different readers. In that case use ONLY the English-labeled column ("Description of Goods" or equivalent) for product_description and DISCARD the other language's column entirely; do not merge "COSSE LUG" (French "COSSE" + English "LUG") or "RACCORD COUDE BACKSHELL" (French "RACCORD COUDE" + English "BACKSHELL") into one string -- the correct value in both cases is just the English word ("LUG", "BACKSHELL").
6. part_number: Part number, catalog number, drawing code, or item code (from "PART NO", "ITEM CODE", etc.). Do NOT put order numbers here.
7. model_or_type: Short model designation, type, or physical/electrical specification only (e.g. "F2", "CLASS 150", "TYPE-B", "B270"). DO NOT copy full product description sentences or text into model_or_type. If there is no distinct short model/type designation, set to null. Do NOT put order numbers here.
8. order_number: Buyer Order No, Purchase Order (PO) number, or Order item reference (e.g. from "(ORDER NO)" or "PO #"). There is no separate field for a PO's date, so if a PO is printed together with its own date (e.g. under a shared "Buyer's Order No. & Date" column, as "446297.1 / 20-APR-26"), keep that number and its date together as-is -- do not split them apart. If a SINGLE line item (per rule 17 below) has more than one PO/order reference tied to it -- e.g. it belongs to a combined document covering two paired invoices, each with its own PO -- include ALL of them here, each kept as its own number+date unit, joined by " / " in the order printed (e.g. "446297.1 / 20-APR-26  /  450044.2 / 15-JUN-26" would be wrong and ambiguous -- instead separate the two units clearly, e.g. "446297.1 (20-APR-26) / 450044.2 (15-JUN-26)"). Never flatten multiple PO+date pairs into one list where it's unclear which date belongs to which PO.
9. hsn_code: ONLY the HSN / SAC / tariff-code column: a 4-, 6- or 8-digit code printed like "8431 49 90" or "84314980". Tables often have a separate short "Item No" / "Line No" column (10, 120, 160...) right next to it -- that is NOT the HSN: never use it, and never join it onto the code. Null if no HSN is printed. If the table prints both an exporter-country tariff column ("HTS") and an "INDIA HTS" column, the HSN is the INDIA HTS value (e.g. "8537.10.10", not the longer "8537.10.9199").
10. quantity: Exact quantity for that specific line item (numeric or null).
11. unit_of_measurement: Unit of measure (e.g. "PC", "PCS", "NOS", "KG", "SET").
12. unit_price: Price / rate per unit for that line item (numeric or null).
13. line_total: Taxable / total amount for that line item (numeric or null).
14. end_use_code: Extract "EU Code" / "End Use Code" ONLY if explicitly printed on the document. If not printed, set to null. NEVER invent codes.
15. fta_code: Free Trade Agreement code ONLY if explicitly claimed. If not claimed, set to null. NEVER put "AD CODE" here.
16. Use JSON null or "" for missing values, NEVER bare unquoted N/A or None.
17. A row starts a NEW line item ONLY if that row has its OWN quantity and amount/rate printed on it. If a later row only adds another reference/invoice/serial number (e.g. in the leftmost marks-and-numbers column) or continues a wrapped value (like an HSN code split across two lines) WITHOUT a quantity and amount of its own, it is a CONTINUATION of the immediately preceding item -- do not create a new line item for it, and do not duplicate the preceding item's quantity/rate/amount onto it. Example: if row 1 shows "INV-A | description | HSN: | 1 | 500.00 | 500.00" and row 2 (directly below, same item block) shows only "INV-B | 84831099" with no quantity/amount, that is ONE line item, not two. This is independent of whether values match -- do NOT merge two rows just because their quantity/rate/amount happen to be equal: if row 1 and row 2 EACH print their own quantity and amount (even identical ones, e.g. two separate parts that are each qty=1 at the same price), they are two distinct line items and must both be kept.
18. Some invoices bundle a PACKING LIST alongside the commercial invoice, on the same or a later page -- a second table repeating the same part number/description with columns like "Net Wt.(Kgs)" / "Gross Wt.(Kgs)" / "Box No." / "Qty per box" / "Sr. No. of Box" instead of a rate and monetary amount. Do NOT treat a row from a packing-list-style table as a new line item, even though it has its own "quantity" and two numbers that LOOK like they could fill unit_price/line_total by position -- a weight in kilograms is not a price. Recognize this by the column headers (Net Wt., Gross Wt., Box No., Pkgs, Qty/per box) rather than by the row's position on the page. If the same part number/description already appears as a line item from the main pricing table, skip its packing-list repetition entirely rather than emitting a second line item with an empty or weight-derived price.
19. This is a SINGLE, non-interactive call -- there is no follow-up turn, no human waiting to say "yes, continue", and no way for you to send a second message. You MUST return the COMPLETE "line_items" array, covering every row on every page, in this ONE response, no matter how many rows there are (tables with 100+ rows are normal for this task). NEVER truncate the array, NEVER summarize or sample only some rows, and NEVER end your response by asking whether to continue, offering to send the rest in batches, or describing what you would do next -- any such text is not valid JSON and breaks the caller, which cannot see or answer it. If the table is very large, work through it efficiently and tersely, but still emit every row's object in the final JSON.
20. packages: THIS ITEM's OWN physical package count (cartons/cases/boxes/pallets/bags/drums -- whichever word this invoice uses), ONLY if a count is printed specifically within this item's own row or block, separately from any other item's count. Confirmed real case: an invoice's two line items each had their own "Total no of packages" line printed right next to that item's own row -- 108 for item 1, 504 for item 2 (packed on different, item-specific pallet ranges) -- and these must stay 108 and 504 on their respective items, NEVER the same number on both, and NEVER their sum (612) on either. Do not confuse this with a single document-wide package count that covers ALL items together (that one is a header-level concept, not this field) -- if the ONLY package count on the invoice is a single whole-shipment figure with no item-specific breakdown, leave this field null on every item rather than guess or copy that one number onto every row. (numeric or null)
21. If the PDF bundles other documents behind the invoice (purchase order, certificates of analysis, a transporter copy of the invoice, terms), take line items ONLY from the invoice's own item table, once; the same line repeated in those documents is not another item. BUT one item table that continues over several pages is ONE table: extract the rows of EVERY page, and do not skip a page because it repeats the invoice header.
22. Copy every digit of quantity, rate and amount as printed; never compute or "correct" a figure; null if not printed.
23. country_of_origin: ONLY a per-row country-of-origin value printed in that row's own column (headed "COO", "Country of Origin" or "Origin"), copied as printed (e.g. "CN", "VN", "MX"). The shipper's / ship-from country is NOT an item's country of origin. Null when the table has no such column.
"""


@_llm_retry
async def _call_header_model(client: AsyncOpenAI, request_kwargs: dict[str, Any]):
    return await client.chat.completions.create(**request_kwargs)


@_llm_retry
async def _call_line_items_model(client: AsyncOpenAI, request_kwargs: dict[str, Any]):
    return await client.chat.completions.create(**request_kwargs)


async def extract_invoice_header(
    invoice_text: str,
    tables_text: str = "",
    model: str = "gpt-4.1-nano",
    api_key: str | None = None,
    feedback: str | None = None,
    previous_result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Pass 1: Extracts document-level header metadata (async).
    Returns (parsed_data, token_usage).

    `feedback` / `previous_result`: only set on a re-extraction attempt
    (see reextraction.py) -- when present, the previous attempt's header
    JSON and a description of which validation rules it failed are
    appended to the user turn, alongside the SAME original invoice text
    used the first time (the model must have the source text in front of
    it to correct anything against -- sending only the previous JSON and
    the complaint would just invite it to guess). The system prompt
    itself is unchanged either way, so a normal (non-retry) call behaves
    exactly as before.
    """
    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    user_content = "INVOICE TEXT:\n" + invoice_text
    if tables_text:
        user_content = "EXTRACTED TABLES:\n" + tables_text + "\n\n" + user_content
    if feedback:
        user_content += (
            "\n\n--- YOUR PREVIOUS EXTRACTION (for reference) ---\n"
            + json.dumps(previous_result or {}, ensure_ascii=False)
            + "\n\n--- VALIDATION FEEDBACK ---\n" + feedback
        )

    request_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": _HEADER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    if not _is_reasoning_model(model):
        request_kwargs["response_format"] = {"type": "json_object"}
    # NOT setting reasoning_effort/verbosity here (unlike the line-items
    # pass below) -- tried "minimal" and "low" on a real invoice
    # (INV10.pdf) whose buyer is only identifiable by inference (no "Bill
    # To"/"Ship To" label at all, just a company name printed under the
    # sender's own letterhead): "minimal" left buyer_name/consignee_name
    # BLANK entirely, and "low" filled buyer_name with "CHILAMPARASANV" --
    # a contact person's name, not the actual buyer company -- where the
    # unrestricted default effort got the correct company name both times.
    # The header pass's fields more often need this kind of inference
    # (unlabeled recipient blocks, inferring country of origin/destination,
    # reconciling conflicting values) than the line-items pass does, so the
    # savings here aren't worth a confirmed, reproducible accuracy
    # regression on exactly the fields this whole pipeline exists to get
    # right. Revisit only with a broader accuracy comparison across many
    # documents, not cost alone.

    logger.info("Calling OpenAI (Pass 1 - Header): model=%s", model)
    resp = await _call_header_model(client, request_kwargs)
    usage = _extract_usage(resp)
    raw = resp.choices[0].message.content
    return _extract_json(raw), usage


async def extract_line_items(
    invoice_text: str,
    tables_text: str = "",
    model: str = "gpt-4.1-nano",
    api_key: str | None = None,
    feedback: str | None = None,
    previous_result: list[dict[str, Any]] | None = None,
    reasoning_effort: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pass 2: Extracts table line items (async).
    Returns (line_items_list, token_usage).

    `feedback` / `previous_result`: see extract_invoice_header's docstring
    -- same mechanism, applied to the line-items pass.

    `reasoning_effort`: reasoning models only; None keeps the cheap
    "minimal" default. See _FINE_RETRY_REASONING_EFFORT for when a
    stronger setting is worth paying for.
    """
    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    user_content = "INVOICE TEXT:\n" + invoice_text
    if tables_text:
        user_content = "EXTRACTED TABLES:\n" + tables_text + "\n\n" + user_content
    if feedback:
        user_content += (
            "\n\n--- YOUR PREVIOUS EXTRACTION (for reference) ---\n"
            + json.dumps({"line_items": previous_result or []}, ensure_ascii=False)
            + "\n\n--- VALIDATION FEEDBACK ---\n" + feedback
        )

    request_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": _LINE_ITEMS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    if not _is_reasoning_model(model):
        request_kwargs["response_format"] = {"type": "json_object"}
    else:
        # See extract_invoice_header's matching comment -- same reasoning:
        # this is table transcription against text already provided, not a
        # task that benefits from GPT-5's (expensive, output-billed) deep
        # reasoning.
        request_kwargs["reasoning_effort"] = reasoning_effort or "minimal"
        request_kwargs["verbosity"] = "low"

    logger.info("Calling OpenAI (Pass 2 - Line Items): model=%s", model)
    resp = await _call_line_items_model(client, request_kwargs)
    usage = _extract_usage(resp)
    raw = resp.choices[0].message.content
    data = _extract_json(raw)

    if isinstance(data, list):
        return data, usage
    if isinstance(data, dict):
        for key in ("line_items", "items", "ITEM", "items_list"):
            if key in data and isinstance(data[key], list):
                return data[key], usage
    return [], usage


# Matches pdf_reader.py's own page-marker line (see extract_full_pdf's
# `f"{'='*35} PAGE {idx} OF {num_pages} (...) {'='*35}"`) -- used only to
# find where one page's text ends and the next begins, not to parse the
# marker's own contents.
_PAGE_MARKER_RE = re.compile(r"^=+\s*PAGE\s+\d+\s+OF\s+\d+", re.MULTILINE)

# Page count above which the line-items pass is split into one LLM call
# per page instead of one call for the whole document. Confirmed real
# case: a single page (~6,000 chars, 21 rows) extracted all 21 rows
# perfectly, with correctly English-only descriptions; the SAME 8-page
# document (~43,000 chars, ~140 rows) in one call returned only 8 rows,
# some with French+English descriptions wrongly concatenated. The model
# loses coverage over a long, highly repetitive table well before any
# hard token limit is reached (finish_reason was "stop", not "length" --
# it isn't being cut off, it's quietly deciding a partial answer is
# enough). Single/few-page invoices -- the overwhelming majority this
# pipeline processes -- are unaffected by this threshold.
_LINE_ITEMS_CHUNK_PAGE_THRESHOLD = 3

# Confirmed real case: a 45-page invoice split into 45 page-chunk calls, all
# fired at once via asyncio.gather with no throttling, saturated the org's
# 200,000 TPM rate limit in a single burst -- every one of tenacity's 4
# retry attempts hit the exact same limit again, since a retry just re-fires
# the whole unthrottled burst rather than spacing it out. A semaphore here
# caps how many page-chunk calls are ever in flight at once, so a large
# document's total token demand gets spread across multiple sequential
# waves instead of arriving as one spike. Left generous enough that it only
# meaningfully changes behavior for documents already past
# _LINE_ITEMS_CHUNK_PAGE_THRESHOLD pages -- a normal short invoice never
# creates enough chunks to hit this cap anyway.
_LINE_ITEMS_CHUNK_CONCURRENCY = 4

# How many per-page chunks (from _split_text_by_page) get grouped into one
# LLM call. >1 trades some of the row-dropping safety margin above for
# fewer calls -- each call re-pays the full system prompt (~2,500 tokens)
# and its own separate reasoning-model invocation overhead, so a 45-page
# document at pages_per_chunk=1 (45 calls) vs =2 (23 calls) roughly halves
# that per-call cost. Kept deliberately small (2, not higher) given the
# confirmed 8-page/~43,000-char failure documented above: that's the
# actual ceiling this constant must stay well under, and 2 pages hasn't
# been independently verified against a table-dense document as part of
# THIS change -- validate row-completeness (item count, not just cost)
# against a known page-dense large invoice before raising this further.
# If any row-dropping reappears at this size, revert to 1 (the original
# per-page behavior).
_LINE_ITEMS_PAGES_PER_CHUNK = 2

# Reasoning effort for the "rows went missing" retry (fine_chunks), which
# only runs after the printed total proved the cheap first pass lost or
# misread rows. Confirmed on two real Flowserve export invoices (38 and 15
# rows, 10 and 5 dense pages, 3 runs each per setting): one page per call at
# "minimal" effort still lost a row or misread an amount in 3 of 6 runs, and
# two pages per call lost or garbled rows in most runs at any effort; one
# page per call at the model's default (medium) effort returned every row
# with an exact total in 6 of 6. "minimal" is kept for the first pass, where
# it is right far more often than not and several times cheaper.
_FINE_RETRY_REASONING_EFFORT = "medium"


def _split_text_by_page(text: str) -> list[str]:
    """
    Splits `text` into one chunk per page, using the "PAGE N OF M"
    markers pdf_reader.py inserts before each page's content. Each
    returned chunk starts with its own page marker line, so the model
    reading it still sees which page it's looking at. Returns `[text]`
    unchanged if fewer than 2 markers are found (single-page document,
    or text from some other source that never carried these markers).
    """
    starts = [m.start() for m in _PAGE_MARKER_RE.finditer(text)]
    if len(starts) <= 1:
        return [text]
    starts.append(len(text))
    return [text[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def _group_pages_into_chunks(pages: list[str], pages_per_chunk: int) -> list[str]:
    """Groups consecutive per-page chunks (from `_split_text_by_page`) into
    larger chunks of `pages_per_chunk` pages each, joined so every page's
    own "PAGE N OF M" marker stays intact and visible to the model. This
    cuts the number of separate LLM calls roughly `pages_per_chunk`-fold
    (see `_LINE_ITEMS_PAGES_PER_CHUNK`) without changing what content the
    model sees overall -- the same total text, just batched into fewer
    calls. `pages_per_chunk <= 1` is a no-op (returns `pages` unchanged)."""
    if pages_per_chunk <= 1:
        return pages
    return ["\n\n".join(pages[i:i + pages_per_chunk]) for i in range(0, len(pages), pages_per_chunk)]


_HEADER_QTY_RE = re.compile(r"\b(?:qty|quantity|qnty)\b", re.IGNORECASE)
_HEADER_ITEM_RE = re.compile(r"\b(?:description|hsn|part\s*no|item)\b", re.IGNORECASE)
_DATA_ROW_RE = re.compile(r"^\s*\d+\s*[│|]")
# A column-header line never carries a decimal amount, a 6+ digit code
# (order no., HSN, part no.) or a dd-MON-yy date; a data row almost always
# does. Confirmed real miss: Flowserve rows start with an order number or a
# date, not "<n> │", so _DATA_ROW_RE alone never fired and the two data rows
# under the header (including a $446.52 item) were copied into every
# continuation page's prompt as "headers" -- the model re-extracted them on
# each page, adding 1-4 phantom rows to a 15-row invoice in every run.
_DATA_LINE_HINT_RE = re.compile(r"\d[\d,]*\.\d{2,}|\b\d{6,}\b|\b\d{1,2}-[A-Za-z]{3}-\d{2,4}\b")


def _table_header_context(page_text: str) -> str:
    """The item table's column-header lines from a page's text (the first
    line naming both a quantity and a description/HSN/part/item column, plus
    the up-to-5 header lines under it that precede the first data row), or
    "" if no such header is found. Continuation pages of a long table often
    don't repeat it, and rows without column names are easy to misread."""
    lines = page_text.splitlines()
    for i, line in enumerate(lines):
        if _HEADER_QTY_RE.search(line) and _HEADER_ITEM_RE.search(line) and not re.match(r"^\s*\d", line):
            block = [line]
            for nxt in lines[i + 1:i + 6]:
                if not nxt.strip():
                    continue
                if _DATA_ROW_RE.match(nxt) or _DATA_LINE_HINT_RE.search(nxt):
                    break
                block.append(nxt)
            return "\n".join(x.rstrip() for x in block)
    return ""


# ── transport documents bundled with the invoice ───────────────────────

_TRANSPORT_DOC_RE = re.compile(r"\bair\s*waybill\b|\bbill\s+of\s+lading\b", re.IGNORECASE)
_TRANSPORT_FORM_RE = re.compile(
    r"shipper'?s\s+(?:name\s+and\s+address|account\s+number)|consignee'?s\s+(?:name\s+and\s+address|account\s+number)"
    r"|nature\s+and\s+quantity\s+of\s+goods|notify\s+part(?:y|ies)",
    re.IGNORECASE,
)


def _is_transport_document_page(page_text: str) -> bool:
    """True for an air waybill / bill of lading form page: it names the
    document type AND carries the carrier form's own box labels (shipper /
    consignee address boxes, "Nature and Quantity of Goods")."""
    return bool(_TRANSPORT_DOC_RE.search(page_text)) and len(_TRANSPORT_FORM_RE.findall(page_text)) >= 2


def _drop_transport_document_pages(invoice_text: str) -> str:
    """The text with air-waybill / bill-of-lading pages removed, for the
    line-items pass only. Confirmed real case: each HYVE file is waybill +
    commercial invoice + packing list, and the waybill's "Nature and
    Quantity of Goods" ("Computer Equipment-454896", its own HS code and
    piece count) was extracted as a line item -- a phantom row beside the
    real one, or in place of it, with the invoice number as the part
    number. A waybill describes the consignment, never the invoice's items.
    Pages are only dropped when at least one other page remains; the header
    pass still sees everything (ports, weights, parties come from there)."""
    pages = _split_text_by_page(invoice_text)
    if len(pages) < 2:
        return invoice_text
    kept = [pg for pg in pages if not _is_transport_document_page(pg)]
    if not kept or len(kept) == len(pages):
        return invoice_text
    logger.info("Line-items pass: skipping %d air-waybill / bill-of-lading page(s)", len(pages) - len(kept))
    return "".join(kept)


async def extract_line_items_chunked(
    invoice_text: str,
    tables_text: str = "",
    model: str = "gpt-4.1-nano",
    api_key: str | None = None,
    feedback: str | None = None,
    previous_result: list[dict[str, Any]] | None = None,
    fine_chunks: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Drop-in replacement for extract_line_items() -- same signature (plus
    `fine_chunks`, below), same
    return shape -- that splits the line-items pass into calls of up to
    _LINE_ITEMS_PAGES_PER_CHUNK pages each for large, multi-page tables,
    then concatenates the results in page order. Falls back to a single
    plain extract_line_items() call for anything at or under
    _LINE_ITEMS_CHUNK_PAGE_THRESHOLD pages, so the overwhelming majority of
    (short) invoices this pipeline sees are completely unaffected -- this
    only changes behavior for the large tables that were actually losing
    rows.

    On a retry (`feedback` set), every chunk is re-run with the SAME
    overall feedback text, rather than trying to map specific validator
    findings -- indexed into the FLAT, already-merged item list -- back
    to the one chunk that produced them. That mapping isn't
    reliable to reconstruct, and isn't needed often enough to be worth
    the complexity: most retryable item-level rules (e.g.
    QTY_X_UNIT_PRICE) describe a self-contained check a chunk can re-verify
    against its own rows regardless of which chunk originally produced
    the flagged index. For the same reason, `previous_result` (the FLAT
    merged list) is accepted for signature compatibility with
    extract_line_items() but not threaded down to individual chunk calls.

    `fine_chunks=True` is the "rows went missing" retry (see
    reextraction.py / the LINE_ITEMS_INCOMPLETE rule): ONE call per page for
    ANY multi-page document, each continuation page prefixed with the item
    table's column headers copied from page 1, and no feedback text (a
    per-page call can't act on a document-wide complaint). Confirmed real
    case: given a 2-page table in a single call, the model returned exactly
    page 1's 14 rows and none of page 2's -- one page per call doesn't give
    it that option. Needs at least 2 pages; otherwise behaves as before.
    """
    invoice_text = _drop_transport_document_pages(invoice_text)
    pages = _split_text_by_page(invoice_text)
    fine = fine_chunks and len(pages) >= 2
    if not fine and len(pages) <= _LINE_ITEMS_CHUNK_PAGE_THRESHOLD:
        return await extract_line_items(
            invoice_text, tables_text=tables_text, model=model, api_key=api_key,
            feedback=feedback, previous_result=previous_result,
        )

    import asyncio
    if fine:
        header_ctx = _table_header_context(pages[0])
        prefix = (
            "COLUMN HEADERS OF THE ITEM TABLE (copied from page 1 as context only -- extract rows "
            "ONLY from the page text below this block, never from the headers themselves. The page "
            "below may contain NO item rows at all (e.g. a final totals, terms or packing page) -- "
            "in that case return an empty line_items list):\n"
            + header_ctx + "\n\n"
        ) if header_ctx else ""
        chunks = [pages[0]] + [prefix + p for p in pages[1:]]
        chunk_feedback = None
        logger.info("Line-items pass (completeness retry): one call per page, %d pages", len(pages))
    else:
        chunks = _group_pages_into_chunks(pages, _LINE_ITEMS_PAGES_PER_CHUNK)
        chunk_feedback = feedback
        logger.info(
            "Line-items pass: %d pages exceeds the %d-page single-call threshold -- "
            "splitting into %d calls of up to %d page(s) each (max %d concurrent)",
            len(pages), _LINE_ITEMS_CHUNK_PAGE_THRESHOLD, len(chunks), _LINE_ITEMS_PAGES_PER_CHUNK,
            _LINE_ITEMS_CHUNK_CONCURRENCY,
        )
    semaphore = asyncio.Semaphore(_LINE_ITEMS_CHUNK_CONCURRENCY)

    chunk_effort = _FINE_RETRY_REASONING_EFFORT if fine else None

    async def _bounded_extract(chunk_text: str):
        async with semaphore:
            return await extract_line_items(
                chunk_text, model=model, api_key=api_key, feedback=chunk_feedback,
                reasoning_effort=chunk_effort,
            )

    page_results = await asyncio.gather(*(_bounded_extract(chunk_text) for chunk_text in chunks))

    items: list[dict[str, Any]] = []
    total_usage = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    for page_items, usage in page_results:
        items.extend(page_items)
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

    return items, total_usage


def _to_float(value: Any) -> float | None:
    """Best-effort float for a possibly-formatted number ("1,234.50", 12,
    None). None means "no usable number" -- never 0 -- including for bools
    and for combined-document values like "4416.00 / 1472.00"."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


# ── OCR-variant duplicate invoice numbers ────────────────────────────────
#
# Confirmed real case (two Hetero invoices): the SAME invoice number is
# printed on every page (commercial invoice, bank-details sheet, packing
# list), and OCR reads it differently from page to page -- "SI3626204603"
# (letter I) on one page, "S13626204603" (digit 1) on another. The model
# took the two spellings for two combined sub-invoices, wrote
# "SI3626204603 / S13626204603" with totals "30875.00 / 30875.00", and the
# validator then summed those into 61,750.00 against a real total of
# 30,875.00 -- a doubled total that exists nowhere in the document.

# Same separator the validator uses for combined documents: a "/" with
# whitespace on at least one side. A bare "/" is part of the number itself
# ("KA/2627/I/033552").
_COMBINED_SEP_RE = re.compile(r"\s+/\s*|\s*/\s+")
# Characters OCR routinely swaps in identifiers: I/L<->1, O<->0.
_OCR_CONFUSABLE = str.maketrans({"I": "1", "L": "1", "O": "0"})


def _split_combined_parts(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    return [p.strip() for p in _COMBINED_SEP_RE.split(str(value)) if p.strip()]


def _ocr_key(s: str) -> str:
    """Canonical form for "is this the same identifier, read two ways":
    upper-cased, non-alphanumerics dropped, OCR-confusable characters
    folded together."""
    return re.sub(r"[^A-Z0-9]", "", s.upper().translate(_OCR_CONFUSABLE))


def collapse_duplicate_invoice_refs(header_raw: dict[str, Any], invoice_text: str) -> list[str]:
    """
    If invoice_number lists several numbers that are really ONE number
    read differently by OCR, collapse them to a single number, and collapse
    the parallel invoice_date / invoice_total lists with it. Returns
    human-readable notes for the output's warnings (empty if nothing
    changed).

    Deliberately conservative -- nothing is collapsed unless EVERY
    duplicate group agrees: the aligned dates and totals of the
    duplicates must be equal. Two genuinely distinct sub-invoices whose
    numbers differ in a real digit are never merged (the canonical keys
    differ), and "A / A" with totals 100 / 200 is left alone for the
    validator to flag rather than guessed at.

    Which spelling survives: both spellings literally occur in the
    document, so this only chooses between values the source contains.
    The one with MORE LETTERS wins ("SI3626204603" over "S13626204603"),
    because OCR turning a letter into a digit/look-alike is far more
    common than the reverse inside a letter-prefixed identifier; ties go to
    the spelling occurring more often in the text, then to the first.
    """
    numbers = _split_combined_parts(header_raw.get("invoice_number"))
    if len(numbers) < 2:
        return []
    keys = [_ocr_key(n) for n in numbers]
    if len(set(keys)) == len(keys):
        return []  # all genuinely distinct sub-invoices

    dates = _split_combined_parts(header_raw.get("invoice_date"))
    totals = _split_combined_parts(header_raw.get("invoice_total"))
    dates_aligned = len(dates) == len(numbers)
    totals_aligned = len(totals) == len(numbers)

    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)

    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        if dates_aligned and len({dates[i] for i in idxs}) > 1:
            return []
        if totals_aligned:
            vals = {_to_float(totals[i]) for i in idxs}
            if len(vals) > 1 or None in vals:
                return []

    def _best_spelling(idxs: list[int]) -> int:
        return max(
            idxs,
            key=lambda i: (
                sum(c.isalpha() for c in numbers[i]),
                invoice_text.count(numbers[i]),
                -i,
            ),
        )

    kept: list[tuple[int, int]] = []  # (index to take the number from, group's first index)
    for idxs in groups.values():
        kept.append((_best_spelling(idxs), idxs[0]))
    kept.sort(key=lambda pair: pair[1])  # first-appearance order

    old = header_raw.get("invoice_number")
    header_raw["invoice_number"] = " / ".join(numbers[best] for best, _ in kept)
    if dates_aligned:
        header_raw["invoice_date"] = " / ".join(dates[first] for _, first in kept)
    if totals_aligned:
        header_raw["invoice_total"] = " / ".join(totals[first] for _, first in kept)
    return [
        f"invoice_number '{old}' collapsed to '{header_raw['invoice_number']}' -- the same "
        f"invoice number read differently by OCR on different pages, not separate invoices"
    ]


# ── Supplier-specific invoice-number formats ─────────────────────────────
#
# Hetero's invoice numbers are always "SI" + 10 digits (e.g. SI3726201912,
# the one Hetero invoice in the sample set with a real text layer). On
# scans, Tesseract can't reliably tell the serif letter I from the digit 1,
# or S from 8/5, so the same number comes out as S13626204505, 813626102057,
# 513626204505 or SI13626102057 (an extra stray "1") -- and OCR settings
# don't fix it (tried: higher oversampling made it worse). Confirmed across
# the whole Hetero batch: every OCR'd result carried a garbled number,
# including those that passed validation, which only checks arithmetic.
#
# This is deliberately SUPPLIER-SPECIFIC, hand-maintained domain knowledge
# (like schema.py's alias tables), approved explicitly as an exception to
# "never repair a value the text doesn't print": it only rewrites the 2-3
# PREFIX characters, never a digit, and only when the 10-digit core is
# actually present in the document text.
_SUPPLIER_INVOICE_FORMATS = (
    {
        "name": "Hetero",
        "supplier": re.compile(r"\bHETER[O0]\b", re.IGNORECASE),
        "valid": re.compile(r"^SI\d{10}$"),
        # S read as S/5/8/B, I read as I/1/l/!/|, optionally followed by one
        # stray extra "1", then the 10-digit core.
        "misread": re.compile(r"^[S5B8][I1l!|]1?(?P<core>\d{10})$", re.IGNORECASE),
        "canonical": "SI{core}",
    },
)


def repair_invoice_number_by_supplier_format(header_raw: dict[str, Any], invoice_text: str) -> list[str]:
    """Repairs an OCR-garbled invoice number for suppliers with a known
    fixed format (see _SUPPLIER_INVOICE_FORMATS). Mutates `header_raw`;
    returns notes for the output's warnings (empty if nothing changed).
    Each number in a " / "-joined list is handled separately; a number
    already in the right format, or one that doesn't fit the known
    misreading pattern, or whose digits aren't in the text, is untouched."""
    supplier = str(header_raw.get("supplier_name") or "")
    fmt = next((f for f in _SUPPLIER_INVOICE_FORMATS if f["supplier"].search(supplier)), None)
    parts = _split_combined_parts(header_raw.get("invoice_number"))
    if fmt is None or not parts:
        return []

    notes: list[str] = []
    repaired: list[str] = []
    for part in parts:
        new = part
        if not fmt["valid"].match(part):
            m = fmt["misread"].match(part)
            if m and m.group("core") in (invoice_text or ""):
                new = fmt["canonical"].format(core=m.group("core"))
        if new != part:
            notes.append(
                f"invoice_number '{part}' repaired to '{new}': {fmt['name']} invoice numbers are "
                f"'SI' + 10 digits and OCR misread the prefix (the digits are unchanged)"
            )
        repaired.append(new)
    if notes:
        header_raw["invoice_number"] = " / ".join(repaired)
    return notes


# ── Line total vs the amount actually printed on the page ───────────────

_NUMBER_TOKEN_RE = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?!\d)")


def _numbers_in_text(text: str) -> set:
    from decimal import Decimal, InvalidOperation
    found = set()
    for tok in _NUMBER_TOKEN_RE.findall(text or ""):
        try:
            found.add(Decimal(tok.replace(",", "")))
        except InvalidOperation:
            pass
    return found


# ── HSN code: the item-number column must not leak into it ──────────────
#
# Confirmed real case (JCB invoices): the table has "Item No | HSN Code"
# side by side. The model put the Item No (10, 120, 160...) into the HSN
# field on 17 of 17 rows of one invoice, and on OCR'd scans the item number
# got fused onto the code ("420 4016 93 40"). Both repairs below use ONLY
# what the page prints -- a leading item number is dropped from a valid HSN
# that is printed in full, and a missing HSN is taken from the same source
# row when exactly one is printed there.

_HSN_SPACED_RE = re.compile(r"(?<![\d.,/-])\d{4} \d{2}(?: \d{2})?(?![\d.,])")
_HSN_PLAIN_RE = re.compile(r"(?<![\d.,/-])\d{8}(?![\d.,])")


def _squash_row(s: str) -> str:
    return re.sub(r"[│|\s]+", " ", s).strip().lower()


def _row_keys(item: dict[str, Any]) -> list[str]:
    """Strings that identify this item's source row: its part number and the
    start of its description (both lower-cased, whitespace-squashed)."""
    keys: list[str] = []
    part = str(item.get("part_number") or "").strip()
    if len(part) >= 4:
        keys.append(part.lower())
    desc = _squash_row(str(item.get("product_description") or ""))
    if len(desc) >= 8:
        keys.append(desc[:20])
    return keys


def _source_row_lines(item: dict[str, Any], invoice_text: str) -> list[str]:
    """The text line(s) that contain this item's part number (or the start of
    its description) -- i.e. its row in the source table. Empty if the item
    has neither, or neither is found."""
    keys = _row_keys(item)
    if not keys:
        return []
    return [ln for ln in (invoice_text or "").splitlines() if any(k in _squash_row(ln) for k in keys)]


_ROW_START_RE = re.compile(r"^\s*\d{1,3}\s*[│|]\s*\d{1,4}\b")
_BLOCK_STOP_RE = re.compile(r"^\s*(?:total|basic|grand|amount|=)", re.IGNORECASE)


def _source_row_blocks(item: dict[str, Any], invoice_text: str) -> list[str]:
    """One block per source line that contains this item's part number (or the
    start of its description); each block is that line plus up to 2
    CONTINUATION lines: on scans the row's price and amounts often wrap onto
    the line underneath ("3 │ 480 ... │ CN" then "22,65 226,50 │ 226,50").
    A continuation stops at the next row (a numbered row start, or any line
    carrying an HSN code -- a wrapped price line never does, a new row
    always does), a blank line, or a totals line, so it never borrows the
    next item's numbers. Used only for number matching -- NOT for the HSN
    lookup, where a following line could belong to the next item."""
    keys = _row_keys(item)
    if not keys:
        return []
    lines = (invoice_text or "").splitlines()
    blocks: list[str] = []
    for i, ln in enumerate(lines):
        if not any(k in _squash_row(ln) for k in keys):
            continue
        block = [ln]
        for nxt in lines[i + 1:i + 3]:
            if (not nxt.strip() or _ROW_START_RE.match(nxt) or _BLOCK_STOP_RE.match(nxt)
                    or _HSN_SPACED_RE.search(nxt) or _HSN_PLAIN_RE.search(nxt)):
                break
            block.append(nxt)
        blocks.append("\n".join(block))
    return blocks


def _source_row_block(item: dict[str, Any], invoice_text: str) -> str:
    return "\n".join(_source_row_blocks(item, invoice_text))


def _hsn_on_source_row(item: dict[str, Any], invoice_text: str) -> str | None:
    """The single HSN-shaped code printed on the source row(s) that contain
    this item's part number (or the start of its description); None if there
    is none or the matching rows disagree."""
    lines = _source_row_lines(item, invoice_text)
    if not lines:
        return None
    spaced: dict[str, str] = {}
    plain: dict[str, str] = {}
    for line in lines:
        for m in _HSN_SPACED_RE.finditer(line):
            spaced.setdefault(m.group(0).replace(" ", ""), m.group(0))
        for m in _HSN_PLAIN_RE.finditer(line):
            plain.setdefault(m.group(0), m.group(0))
    found = spaced or plain
    return next(iter(found.values())) if len(found) == 1 else None


def _repair_hsn_code(item: dict[str, Any], invoice_text: str, row_no: int) -> str | None:
    raw = item.get("hsn_code")
    if raw is None or raw == "" or isinstance(raw, bool):
        return None
    s = re.sub(r"\s+", " ", re.sub(r"[^\d ]", " ", str(raw))).strip()
    digits = s.replace(" ", "")
    if not digits or len(digits) in (4, 6, 8):
        return None  # nothing to repair, or already shaped like an HSN
    tokens = s.split(" ")
    # (a) a 1-4 digit item number fused onto a valid HSN that is printed in full
    if len(tokens) >= 2 and len(tokens[0]) <= 4:
        rest = " ".join(tokens[1:])
        if len(rest.replace(" ", "")) in (4, 6, 8) and rest in re.sub(r"\s+", " ", invoice_text or ""):
            item["hsn_code"] = rest
            return (f"[line_items][row {row_no}] hsn_code '{raw}' -> '{rest}': a leading item number "
                    f"was fused onto the HSN code printed in the source")
    # (b) 1-3 digits can't be an HSN: use the one printed on this item's row
    if len(digits) <= 3:
        found = _hsn_on_source_row(item, invoice_text)
        if found:
            item["hsn_code"] = found
            return (f"[line_items][row {row_no}] hsn_code '{raw}' is an item number, not an HSN; replaced "
                    f"with '{found}', the only HSN printed on that item's row")
    # (c) a 10-digit exporter-country tariff code ("HTS 8517.62.0000") where the
    # same printed line also carries the 8-digit INDIA HTS under it ("8517.62.90")
    if len(digits) == 10:
        india = {m for ln in (invoice_text or "").splitlines() if str(raw).strip() in ln
                 for m in re.findall(r"(?<![\d.])\d{4}\.\d{2}\.\d{2}(?![\d.])", ln)
                 if m.replace(".", "")[:6] == digits[:6]}
        if len(india) == 1:
            found = next(iter(india))
            item["hsn_code"] = found
            return (f"[line_items][row {row_no}] hsn_code '{raw}' has 10 digits; the same line prints the "
                    f"8-digit INDIA HTS '{found}', which is the Indian HSN; replaced")
    return None


# ── Unit price taken for the line amount (shifted price columns) ────────
#
# Confirmed real case (JCB MH2706215472, wrong on EVERY run and every prompt
# wording tried): the row prints "Sales Price Per Unit 1,750.29 | Basic Value
# 3,500.58 | Total 3,500.58" for qty 2. The model took 1,750.29 -- the UNIT
# price -- as the line amount, and then wrote a unit price of 875.145 that is
# printed nowhere. Both halves are wrong, yet qty x price == amount, so the
# arithmetic checks pass the pair; only the printed total exposes it.

def _to_decimal(v: Any):
    from decimal import Decimal
    f = _to_float(v)
    return None if f is None else Decimal(str(f))


# A MONEY amount: a number whose last separator is followed by EXACTLY two
# digits, in either convention -- "1,750.29" / "3,500.58" (decimal point) or
# "22,65" / "1.803,21" (decimal comma, as printed on several scanned
# invoices). Weights (3 decimals: "48,700", "5.210"), quantities, PO numbers,
# HSN fragments and dates deliberately don't match.
_MONEY_RE = re.compile(r"(?<![\d.,])\d[\d.,]*[.,]\d{2}(?!\d|[.,]\d)")


def _money_counts(text: str):
    """How many times each MONEY amount occurs in `text` (Decimal -> count),
    reading both decimal-point and decimal-comma formats."""
    from collections import Counter
    from decimal import Decimal, InvalidOperation
    counts: Counter = Counter()
    for tok in _MONEY_RE.findall(text or ""):
        try:
            counts[Decimal(re.sub(r"[.,]", "", tok[:-3]) + "." + tok[-2:])] += 1
        except InvalidOperation:
            pass
    return counts


def _repair_shifted_price_columns(item: dict[str, Any], invoice_text: str, row_no: int) -> str | None:
    """Fixes a unit price / line amount the model took from the WRONG column
    -- the unit price as the amount, the Basic Value as the unit price, a
    weight as the price -- using only numbers printed on that item's own row.
    Applies only when quantity > 1 and the extracted price and amount are
    NOT both printed on the row, and the row has exactly ONE printed
    (unit price, amount) pair with amount = quantity x unit price and the
    amount printed twice. Then both fields are set to that pair; nothing is
    computed that the row doesn't already print.

    Real cases (JCB): 215472 (price 875.145 invented, 1,750.29 taken as the
    amount), 214961 row 22 (Basic Value 48.92 taken as the unit price, 97.84
    derived), 220038 rows 4 and 9 (a weight taken as the price)."""
    from decimal import Decimal
    qty, price, total = _to_decimal(item.get("quantity")), _to_decimal(item.get("unit_price")), _to_decimal(item.get("line_total"))
    if qty is None or price is None or total is None or qty <= 0:
        return None
    blocks = _source_row_blocks(item, invoice_text)
    if not blocks:
        return None
    # Each source row is judged on its OWN numbers, never pooled with another
    # row's. The same part number / description start is often printed on
    # several rows (same part on different POs), and two unrelated rows can
    # print the same amount -- pooled, that looks like one row's "Basic Value
    # + Total". Confirmed real case (Nash 2926425558): CORRECT rows were
    # "repaired" to another row's 24 x 27.00 = 648.00, over-counting the invoice.
    decimal_one_cent = Decimal("0.01")
    per_block = [(blk, _money_counts(blk)) for blk in blocks]
    if any(price in c and total in c and abs(qty * price - total) <= decimal_one_cent for _, c in per_block):
        return None  # both printed on a row as money AND consistent -- nothing to second-guess
    # The printed (unit price, amount) pairs: amount = quantity x unit price,
    # the amount printed at least TWICE on the row (invoices repeat the line
    # amount as "Basic Value" and "Total"; a total WEIGHT, the other
    # qty-multiple on these rows, prints only once), unit price printed too.
    # The unit price counts as printed if it appears as money, OR as bare
    # digits with the separator lost to OCR ("534" for 5.34) -- but only
    # alongside an amount that is itself printed twice and equals quantity x
    # that price, so the arithmetic confirms the reading.
    def _price_printed(p, counts, block) -> bool:
        if p in counts:
            return True
        digits = str((p * 100).quantize(Decimal("1")))
        return re.search(r"(?<![\d.,])" + re.escape(digits) + r"(?![\d.,])", block) is not None

    pairs = {
        (p, a)
        for block, counts in per_block
        for a, n in counts.items() if n >= 2
        for p in [(a / qty).quantize(decimal_one_cent)]
        # (for quantity 1 the unit price IS the amount, so p == a is expected)
        if (p != a or qty == 1) and (p * qty).quantize(decimal_one_cent) == a and _price_printed(p, counts, block)
    }
    if len(pairs) != 1:
        return None
    new_price, new_total = next(iter(pairs))
    if float(new_price) == float(price) and float(new_total) == float(total):
        return None  # already what the row prints -- nothing to change or report
    item["unit_price"], item["line_total"] = float(new_price), float(new_total)
    return (f"[line_items][row {row_no}] unit_price {price} / line_total {total} are not both printed on the "
            f"row; the row prints quantity {qty} x {new_price} = {new_total} (the amount appears twice, as "
            f"Basic Value and Total); corrected to unit_price {new_price}, line_total {new_total}")


# ── charge lines (freight, insurance, ...) are not line items ──────────

_CHARGE_ROW_RE = re.compile(
    r"^(?:(?:AIR|SEA|OCEAN|INLAND)\s+)?(?:FREIGHT|SHIPPING|INSURANCE|HANDLING|PACKING|PACKAGING|FORWARDING)"
    r"(?:\s+(?:CHARGES?|COST|COSTS|FEES?))?$"
)


def _drop_charge_rows(items_raw: list[dict[str, Any]], invoice_text: str) -> list[str]:
    """
    Removes a "line item" that is really a freight/insurance/handling charge
    printed once under the table, e.g. "FREIGHT:   INR 487.42" between the
    last item row and "Total Value". Confirmed real case: the model returned
    it as item 2 with no quantity (REQUIRED_FIELD_MISSING), or -- when the
    text layer had split the amount -- as "6 x 74.28".

    The row goes only if its WHOLE description is a charge label AND the
    source has a line that starts with that label and carries exactly one
    money amount. An item-table row has several amounts (unit price, total)
    and starts with its own SKU/serial, so it never matches. Nothing is
    added or recomputed: the charge stays on the page, and the validator
    reconciles line sum + printed freight against the printed total.
    Mutates `items_raw` in place; returns notes for the output's warnings.
    """
    notes: list[str] = []
    keep: list[dict[str, Any]] = []
    source_lines = [re.sub(r"[│" + chr(92) + "s]+", " ", ln).strip() for ln in (invoice_text or "").splitlines()]
    for idx, it in enumerate(items_raw, start=1):
        desc = re.sub(r"[^A-Za-z ]+", " ", str(it.get("product_description") or "")) if isinstance(it, dict) else ""
        desc = re.sub(r"\s+", " ", desc).strip().upper()
        if not _CHARGE_ROW_RE.match(desc):
            keep.append(it)
            continue
        first_word = desc.split()[0]
        printed_once = [
            ln for ln in source_lines
            if re.match(r"(?i)^" + re.escape(first_word) + r"\b", ln) and len(_MONEY_RE.findall(ln)) == 1
        ]
        if printed_once:
            notes.append(
                f"[line_items][row {idx}] '{desc.title()}' is a charge printed once under the table "
                f"('{printed_once[0][:60]}'), not an item; removed"
            )
        else:
            keep.append(it)
    if len(keep) != len(items_raw):
        items_raw[:] = keep
    return notes


def apply_line_item_guardrails(items_raw: list[dict[str, Any]] | None, invoice_text: str) -> list[str]:
    """
    Repairs a line_total the model MIS-COPIED, only when the page itself
    proves the right value. Confirmed real case: the printed amount was
    "30,875.00" (475 KG x 65.00), the OCR text had it right, and the model
    returned 30775 -- one digit off.

    A line_total is replaced by qty x unit_price ONLY IF all of these hold:
      - qty x unit_price != the extracted line_total (beyond a cent),
      - the extracted line_total does NOT occur anywhere in the source
        text (so it isn't a figure the document itself prints -- a genuine
        printed mismatch is left for the validator to flag), and
      - qty x unit_price DOES occur verbatim in the source text.
    So the replacement is a value the document literally prints, never one
    computed out of thin air; a missing line_total is never filled in.
    Mutates `items_raw`; returns notes for the output's warnings.
    """
    from decimal import Decimal
    if not isinstance(items_raw, list) or not items_raw:
        return []

    def _dec(v: Any):
        f = _to_float(v)
        return None if f is None else Decimal(str(f))

    printed = _numbers_in_text(invoice_text)
    notes: list[str] = _drop_charge_rows(items_raw, invoice_text)
    for idx, it in enumerate(items_raw, start=1):
        if not isinstance(it, dict):
            continue
        hsn_note = _repair_hsn_code(it, invoice_text, idx)
        if hsn_note:
            notes.append(hsn_note)
        shift_note = _repair_shifted_price_columns(it, invoice_text, idx)
        if shift_note:
            notes.append(shift_note)
        qty, price, total = _dec(it.get("quantity")), _dec(it.get("unit_price")), _dec(it.get("line_total"))
        if qty is None or price is None or total is None or qty <= 0 or price <= 0:
            continue
        expected = (qty * price).quantize(Decimal("0.01"))
        if abs(expected - total) <= Decimal("0.01") or total in printed:
            continue
        if expected in printed:
            it["line_total"] = float(expected)
            notes.append(
                f"[line_items][row {idx}] line_total {total} is not printed anywhere on the "
                f"document; replaced with {expected} (= quantity {qty} x unit price {price}), "
                f"which IS printed"
            )
    return notes


def apply_header_guardrails(
    header_raw: dict[str, Any] | None,
    items_raw: list[dict[str, Any]] | None,
    invoice_text: str,
    shipment_type: str | None = None,
    notes: list[str] | None = None,
) -> None:
    """
    Deterministic, source-text-anchored corrections applied to a header
    extraction after the LLM call returns -- belt-and-suspenders checks
    for confirmed real mistakes the prompt alone doesn't reliably avoid
    on its own. Mutates `header_raw` in place; a no-op if it isn't a dict.

    `shipment_type` gates the export-only country_of_origin correction
    below; anything other than "export" (including None) skips it.

    `notes`, if given, is appended to with a human-readable line for any
    correction worth surfacing in the output's warnings.

    Called from reextraction.py's retry loop -- the actual production
    entry point every real run goes through. This function used to be
    inline code living only inside a since-removed extract_fields()
    convenience wrapper, which reextraction.py never called -- none of
    these guardrails were actually reachable from production until this
    was pulled out into its own function reextraction.py invokes directly.
    """
    if not isinstance(header_raw, dict):
        return
    low_text = invoice_text.lower()

    # Guardrail: one invoice number, spelled two ways by OCR on different
    # pages, is not two combined invoices (see collapse_duplicate_invoice_refs).
    # For suppliers with a known invoice-number format, undo OCR misreads of
    # the prefix FIRST: spellings that differ by S/5/8 (both seen on real
    # pages) aren't merged by the generic I/1/O/0 collapse below, but become
    # identical once each is repaired to the canonical format.
    collapse_notes = repair_invoice_number_by_supplier_format(header_raw, invoice_text)
    collapse_notes += collapse_duplicate_invoice_refs(header_raw, invoice_text)
    if notes is not None:
        notes.extend(collapse_notes)

    # Guardrail: the model is asked never to guess net_realisable_amount,
    # but as a belt-and-suspenders check, force it to None unless the
    # source text actually contains one of the phrases that field means.
    if "net realis" not in low_text and "net charge" not in low_text and "realisable" not in low_text:
        if header_raw.get("net_realisable_amount") is not None:
            header_raw["net_realisable_amount"] = None

    # Guardrail: invoice_total sometimes gets set to a "Total F.O.B Value"
    # figure despite the prompt rule against it. Confirmed on a real
    # invoice: that value was in INR (== gross line-item total x the
    # printed USD/INR exchange rate) while `currency` was correctly "USD"
    # -- i.e. it's not just the wrong number, it's the wrong currency's
    # number under the right currency's label. If invoice_total matches a
    # "F.O.B Value" line in the source text AND there's evidence that line
    # is in a different currency, prefer net_realisable_amount (the
    # confirmed actual chargeable amount) if we have it, else drop
    # invoice_total rather than keep a silently-wrong total.
    #
    # "Matches an FOB Value line" alone is NOT evidence of a currency
    # mismatch: on an invoice whose own trade term is FOB, the FOB value
    # legitimately equals the invoice total, in the same currency, and the
    # old match-only rule nulled that correct total. Evidence required
    # (any one):
    #   - the FOB line itself is labeled INR / Rs / rupees / the rupee
    #     sign, while the invoice currency isn't INR; or
    #   - every line item has an amount and their sum contradicts
    #     invoice_total (so invoice_total is demonstrably not the amount
    #     those items add up to -- the confirmed case: items summed to the
    #     USD total, invoice_total held the larger INR FOB figure).
    # With neither, invoice_total is left alone: a possibly-wrong-currency
    # total is still caught downstream by HEADER_LINE_TOTAL, whereas a
    # deleted correct one is silent data loss.
    invoice_total = _to_float(header_raw.get("invoice_total"))
    if invoice_total is not None and (header_raw.get("currency") or "").strip().upper() != "INR":
        items_for_sum = items_raw if isinstance(items_raw, list) else []
        line_amounts = [_to_float(it.get("line_total")) for it in items_for_sum if isinstance(it, dict)]
        line_sum_known = bool(line_amounts) and all(a is not None for a in line_amounts)
        line_sum = sum(line_amounts) if line_sum_known else None
        items_contradict_total = (
            line_sum is not None and abs(line_sum - invoice_total) > max(1.0, invoice_total * 0.01)
        )

        fob_value_re = (
            r"f\.?\s*o\.?\s*b\.?\s*value"
            r"(?:\s*\(?\s*(?:in\s+)?(?:inr|rs\.?|₹|usd|eur|gbp)\s*\)?)?"  # optional "(INR)" / "in INR"
            r"\s*[:\-]?\s*(?:inr|rs\.?|₹)?\s*([\d,]+\.?\d*)"
        )
        for fob_match in re.finditer(fob_value_re, low_text):
            fob_value = _to_float(fob_match.group(1))
            if not fob_value or fob_value <= 0:
                continue
            if abs(invoice_total - fob_value) > max(1.0, fob_value * 0.01):
                continue
            line_start = low_text.rfind("\n", 0, fob_match.start()) + 1
            line_end = low_text.find("\n", fob_match.end())
            fob_line = low_text[line_start:line_end if line_end != -1 else len(low_text)]
            fob_line_is_inr = re.search(r"\binr\b|\brs\b\.?|\brupees?\b|₹", fob_line) is not None
            if fob_line_is_inr or items_contradict_total:
                net_realisable = header_raw.get("net_realisable_amount")
                header_raw["invoice_total"] = net_realisable if net_realisable is not None else None
            break

    # Guardrail: total_packages sometimes comes back holding the line-item
    # piece quantity itself instead of the actual, much smaller package
    # count -- confirmed on real invoices, where the item table's own
    # "TOTAL" row prints them as two separate numbers in a fixed order
    # ("TOTAL   1        336": 1 is the package count, 336 the quantity),
    # yet the model returned 336, or -- when that row's package-count
    # position is genuinely blank, nothing printed between "TOTAL" and the
    # quantity figure -- guessed a small number anyway instead of leaving
    # it null as instructed. The prompt above was hardened with this exact
    # pattern and fixed most cases on its own, but "notice a field is
    # blank and don't fill it" is a harder compliance bar than "copy this
    # printed number", so this re-derives the true value directly from the
    # source text as a deterministic backstop: anchor on the line-item
    # quantity (already known-good) to find its own "TOTAL" row, and take
    # whichever number printed immediately before it there -- if none of
    # the matching TOTAL rows has a number before the quantity, the
    # package count is genuinely blank on this invoice and gets nulled
    # rather than left as a guess.
    items_for_qty = items_raw if isinstance(items_raw, list) else []
    total_qty = sum(
        q for it in items_for_qty
        if isinstance(q := it.get("quantity"), (int, float))
    ) if items_for_qty else None
    if total_qty:
        candidates = []
        for ln in invoice_text.splitlines():
            if not re.match(r"^\s*TOTAL\b", ln, re.IGNORECASE):
                continue
            nums = [float(n) if "." in n else int(n) for n in re.findall(r"\d+(?:\.\d+)?", ln)]
            if total_qty not in nums:
                continue
            candidates.append(nums)
        with_pkgs = [nums for nums in candidates if nums.index(total_qty) > 0]
        if with_pkgs:
            header_raw["total_packages"] = with_pkgs[0][0]
        elif candidates:
            header_raw["total_packages"] = None

    # Fallback for invoices with no tabular "TOTAL <pkgs> <qty>" row at
    # all -- the real package count instead sits inside a narrative
    # packing-list sentence describing how the quantity was packed,
    # e.g. "Four Nos Of Pallet Containing 3840 Nos (20 Nos Each in 192
    # carton boxes)". There, 3840 is the quantity (already correct),
    # 4 is a PALLET count -- a higher-level handling/consolidation unit
    # that customs' "total_packages" does NOT mean -- and 192, the
    # number sitting immediately before the actual packing-unit word
    # (carton/case/box/package/ctn/bag/drum/crate -- deliberately
    # NOT "pallet"), is the true package count. Confirmed real miss:
    # the model copied the 3840 quantity into total_packages instead.
    # Only fires when the TOTAL-row check above found nothing AND the
    # model's own total_packages still looks like that exact bug
    # (missing, or equal to the quantity) -- an already-different,
    # plausible value is left alone.
    if total_qty:
        current = header_raw.get("total_packages")
        looks_wrong = current in (None, "") or (
            isinstance(current, (int, float)) and float(current) == float(total_qty)
        )
        if looks_wrong:
            unit_matches = re.findall(
                r"(\d[\d,]*)\s*(?:nos\.?)?\s*(?:cartons?|ctns?|cases?|boxes?|"
                r"packages?|pkgs?|crates?|bags?|drums?)\b",
                invoice_text, re.IGNORECASE,
            )
            pkg_candidates = {
                int(n) for m in unit_matches
                if (n := m.replace(",", "")).isdigit()
            }
            pkg_candidates = {c for c in pkg_candidates if 0 < c < total_qty}
            if len(pkg_candidates) == 1:
                header_raw["total_packages"] = pkg_candidates.pop()

    # Guardrail: country_of_origin sometimes comes back as the DESTINATION
    # country instead of the exporter's own country (confirmed on real
    # invoices, all from the same Indian exporter: country_of_origin came
    # back "Germany" / "USA" -- the ship-to country -- while the source
    # text's own, separately-labeled "Country of Origin of Goods" field
    # names India throughout). This only fires in the two structurally
    # unambiguous cases -- it never overrides a plausible value:
    #   1. origin and destination came back identical (goods can't
    #      originate from and be shipped to the same country)
    #   2. origin came back matching what destination should be, i.e. the
    #      two got swapped
    # In both cases the supplier's own country is used, since an invoice's
    # exporter is who the goods originate from unless a transhipment /
    # re-export is separately indicated (transit_country), which neither
    # of these two cases claims.
    #
    # Export only: on an import the destination is ALWAYS India, so case 2
    # fires on every import whose supplier address merely mentions India
    # (an India liaison office, a reseller) and overwrote a real foreign
    # origin with "IN". "INDIA" is also matched as a whole word, so a US
    # supplier in Indiana / Indianapolis doesn't count as Indian.
    if shipment_type != "export":
        return

    def _country_norm(s: Any) -> str:
        s = str(s or "").strip().upper()
        if s in ("INDIA", "IND", "IN"):
            return "IN"
        return s

    supplier_addr = str(header_raw.get("supplier_address") or "").upper()
    origin = _country_norm(header_raw.get("country_of_origin"))
    dest = _country_norm(header_raw.get("country_of_destination"))
    supplier_is_indian = re.search(r"\bINDIA\b", supplier_addr) is not None
    if supplier_is_indian and origin and origin != "IN" and (origin == dest or dest == "IN"):
        header_raw["country_of_origin"] = "IN"

