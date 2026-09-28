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
_USD_TO_INR = 85.0


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
- invoice_number: Invoice / Bill number ONLY (string or null). Some invoices print the number and date together under one shared column/label like "Invoice No. & Date" (e.g. "KA/2627/I/016909/15-JUN-26"). If so, that trailing date-like token is NOT part of the invoice number -- strip it here and put it in invoice_date instead, so invoice_number for that example is just "KA/2627/I/016909". If the document genuinely states MORE THAN ONE invoice number (e.g. a single combined document covering two paired invoice numbers, such as "KA/2627/I/016909" and "KA/2627/I/016910" both printed in the Invoice No. field), include ALL of them here (each with its own date suffix already stripped) joined by " / " in the order printed -- do NOT pick only one and silently drop the rest.
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

- currency: Invoice Currency code (e.g. "USD", "EUR", "INR", "GBP") (string or null)
- invoice_total: The actual amount chargeable/payable to the buyer, in the SAME currency you put in `currency` above (numeric or null). Prefer a line explicitly labeled "Amount Chargeable", "Grand Total", or "Total Invoice Value"; otherwise use the sum of the line-item totals. CRITICAL: NEVER use a "Total F.O.B Value" / "FOB Value" line for this field. On Indian export invoices that value is routinely printed in INR for customs/RBI declaration purposes even when the invoice's trade currency is USD/EUR/etc -- using it here silently mixes a rupee figure into a field tagged with the wrong currency. If the only total-like number you can find is explicitly labeled "F.O.B Value", leave invoice_total null rather than use it. CRITICAL -- if invoice_number above holds MORE THAN ONE invoice number (a combined document): do NOT sum the invoices' totals into one number. Confirmed real case: two FULLY SEPARATE, independently-numbered invoices (each its own header, its own items, its own explicitly printed "TOTAL INVOICE AMOUNT" line -- 4,416.00 for one, 1,472.00 for the other) were bundled into a single PDF; the correct invoice_total is "4416.00 / 1472.00", NOT "5888" (their sum) -- 5888 is not a real amount printed anywhere and doesn't belong to either invoice. Instead, find EACH sub-invoice's own explicitly printed total (same labels as above: "Amount Chargeable", "Grand Total", "Total Invoice Value", "Total Invoice Amount") and join them with " / " in the SAME order as their invoice numbers in invoice_number, so total N corresponds to invoice number N -- exactly like invoice_date does. Only fall back to summing line items for a given sub-invoice if that sub-invoice has no total of its own explicitly printed.
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
      "packages": null
    }
  ]
}

CRITICAL RULES FOR LINE ITEMS:
1. Extract EVERY distinct item row individually across all pages. Do NOT miss any rows and NEVER collapse multiple items into one!
2. Return flat primitive values directly (strings, numbers, null). DO NOT wrap values in {"value": ..., "confidence": ...}.
3. item_ser_no: Extract the EXACT Item Serial / Line Number as printed in the invoice table (e.g. 1, 2, 3... or 10, 20, 30... or 0010, 0020...). If no SNo column is printed on the invoice, set to null. Confirmed real case: an invoice's items were grouped under repeating section headers like "PO No 240300" / "PO No 240400", each followed by a handful of item rows with columns Part No / Description / Qty / Uom / Unit / Amount -- there was no serial-number column anywhere. A "PO No" (or similar) heading that groups rows underneath it is NOT an SNo column, and the row's position under that heading is NOT a serial number either -- do NOT invent your own row-count that increments within each group and resets to 1 at the next heading. In that case item_ser_no must be null on every single row; the caller assigns a real, non-resetting sequence number afterward.
4. Do NOT include subtotal, deduction, or total rows (e.g. "TOTAL", "SUBTOTAL", "LESS: RAW MATERIALS", "NET REALISABLE") as line items.
5. product_description: The commercial item name (e.g. "ROTOR SHAFT ASY", "TO-5 COILS", "ELECTRONIC COMPONENTS"). Omit generic tariff chapter descriptions. The description column frequently WRAPS across multiple physical rows -- both ABOVE and BELOW the specific row that carries the item's own quantity/rate/amount (per rule 17). Read the description column's text on every row belonging to this item (the row with the quantity/rate/amount, plus any row directly above or below it that has no quantity/rate/amount of its own but has description-column text) and CONCATENATE all of it in reading order into one complete description -- do not take only the fragment that happens to share a row with the quantity/rate/amount. Example: "Discovery" on the row above, "IQ 2Ring-Gen2 India" on the row with the quantity/rate/amount -> full description is "Discovery IQ 2Ring-Gen2 India". Example: "AXIAL HEADHOLDER FOR HSA AND NP" on rows above the quantity/rate/amount row, "TABLES-RoHS" on the row below it -> full description is "AXIAL HEADHOLDER FOR HSA AND NP TABLES-RoHS". IMPORTANT -- this vertical (above/below) concatenation is NEVER a license to concatenate HORIZONTALLY across a "│" column boundary: some invoices print the description TWICE side by side as two SEPARATE, fully-labeled columns in different languages (e.g. "Description Produit" in French immediately followed by "Description of Goods" in English, each its own column, separated by "│"). That is not a wrapped/continued value -- it is the SAME field printed twice for two different readers. In that case use ONLY the English-labeled column ("Description of Goods" or equivalent) for product_description and DISCARD the other language's column entirely; do not merge "COSSE LUG" (French "COSSE" + English "LUG") or "RACCORD COUDE BACKSHELL" (French "RACCORD COUDE" + English "BACKSHELL") into one string -- the correct value in both cases is just the English word ("LUG", "BACKSHELL").
6. part_number: Part number, catalog number, drawing code, or item code (from "PART NO", "ITEM CODE", etc.). Do NOT put order numbers here.
7. model_or_type: Short model designation, type, or physical/electrical specification only (e.g. "F2", "CLASS 150", "TYPE-B", "B270"). DO NOT copy full product description sentences or text into model_or_type. If there is no distinct short model/type designation, set to null. Do NOT put order numbers here.
8. order_number: Buyer Order No, Purchase Order (PO) number, or Order item reference (e.g. from "(ORDER NO)" or "PO #"). There is no separate field for a PO's date, so if a PO is printed together with its own date (e.g. under a shared "Buyer's Order No. & Date" column, as "446297.1 / 20-APR-26"), keep that number and its date together as-is -- do not split them apart. If a SINGLE line item (per rule 17 below) has more than one PO/order reference tied to it -- e.g. it belongs to a combined document covering two paired invoices, each with its own PO -- include ALL of them here, each kept as its own number+date unit, joined by " / " in the order printed (e.g. "446297.1 / 20-APR-26  /  450044.2 / 15-JUN-26" would be wrong and ambiguous -- instead separate the two units clearly, e.g. "446297.1 (20-APR-26) / 450044.2 (15-JUN-26)"). Never flatten multiple PO+date pairs into one list where it's unclear which date belongs to which PO.
9. hsn_code: HSN / SAC / Tariff code (numeric string, e.g. "84831099", "85389000").
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
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pass 2: Extracts table line items (async).
    Returns (line_items_list, token_usage).

    `feedback` / `previous_result`: see extract_invoice_header's docstring
    -- same mechanism, applied to the line-items pass.
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


async def extract_line_items_chunked(
    invoice_text: str,
    tables_text: str = "",
    model: str = "gpt-4.1-nano",
    api_key: str | None = None,
    feedback: str | None = None,
    previous_result: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Drop-in replacement for extract_line_items() -- same signature, same
    return shape -- that splits the line-items pass into one call per
    page for large, multi-page tables, then concatenates the results in
    page order. Falls back to a single plain extract_line_items() call
    for anything at or under _LINE_ITEMS_CHUNK_PAGE_THRESHOLD pages, so
    the overwhelming majority of (short) invoices this pipeline sees are
    completely unaffected -- this only changes behavior for the large
    tables that were actually losing rows.

    On a retry (`feedback` set), every page chunk is re-run with the SAME
    overall feedback text, rather than trying to map specific validator
    findings -- indexed into the FLAT, already-merged item list -- back
    to the one page chunk that produced them. That mapping isn't
    reliable to reconstruct, and isn't needed often enough to be worth
    the complexity: most retryable item-level rules (e.g.
    QTY_X_UNIT_PRICE) describe a self-contained check a chunk can re-verify
    against its own rows regardless of which chunk originally produced
    the flagged index. For the same reason, `previous_result` (the FLAT
    merged list) is accepted for signature compatibility with
    extract_line_items() but not threaded down to individual chunk calls.
    """
    pages = _split_text_by_page(invoice_text)
    if len(pages) <= _LINE_ITEMS_CHUNK_PAGE_THRESHOLD:
        return await extract_line_items(
            invoice_text, tables_text=tables_text, model=model, api_key=api_key,
            feedback=feedback, previous_result=previous_result,
        )

    import asyncio
    logger.info(
        "Line-items pass: %d pages exceeds the %d-page single-call threshold -- "
        "splitting into %d per-page calls",
        len(pages), _LINE_ITEMS_CHUNK_PAGE_THRESHOLD, len(pages),
    )
    page_results = await asyncio.gather(*(
        extract_line_items(page_text, model=model, api_key=api_key, feedback=feedback)
        for page_text in pages
    ))

    items: list[dict[str, Any]] = []
    total_usage = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    for page_items, usage in page_results:
        items.extend(page_items)
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

    return items, total_usage


def apply_header_guardrails(
    header_raw: dict[str, Any] | None,
    items_raw: list[dict[str, Any]] | None,
    invoice_text: str,
) -> None:
    """
    Deterministic, source-text-anchored corrections applied to a header
    extraction after the LLM call returns -- belt-and-suspenders checks
    for confirmed real mistakes the prompt alone doesn't reliably avoid
    on its own. Mutates `header_raw` in place; a no-op if it isn't a dict.

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
    # "F.O.B Value" line in the source text, prefer net_realisable_amount
    # (the confirmed actual chargeable amount) if we have it, else drop
    # invoice_total rather than keep a silently-wrong total.
    if header_raw.get("invoice_total") is not None:
        fob_match = re.search(r"f\.?\s*o\.?\s*b\.?\s*value\s*[:\-]?\s*([\d,]+\.?\d*)", low_text)
        if fob_match:
            try:
                fob_value = float(fob_match.group(1).replace(",", ""))
                invoice_total = float(header_raw["invoice_total"])
                if fob_value > 0 and abs(invoice_total - fob_value) <= max(1.0, fob_value * 0.01):
                    net_realisable = header_raw.get("net_realisable_amount")
                    header_raw["invoice_total"] = net_realisable if net_realisable is not None else None
            except (TypeError, ValueError):
                pass

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
    def _country_norm(s: Any) -> str:
        s = str(s or "").strip().upper()
        if s in ("INDIA", "IND", "IN"):
            return "IN"
        return s

    supplier_addr = str(header_raw.get("supplier_address") or "").upper()
    origin = _country_norm(header_raw.get("country_of_origin"))
    dest = _country_norm(header_raw.get("country_of_destination"))
    supplier_is_indian = "INDIA" in supplier_addr
    if supplier_is_indian and origin and origin != "IN" and (origin == dest or dest == "IN"):
        header_raw["country_of_origin"] = "IN"

