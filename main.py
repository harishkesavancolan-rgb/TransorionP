"""
main.py
--------
CLI entry point for extracting a single invoice PDF.

Usage:
    export OPENAI_API_KEY=sk-...
    python main.py path/to/invoice.pdf --out out.json

By default the invoice's shipment type (import vs. export) is auto-detected
from its text (see detect_type.py) and the matching template is loaded
automatically from templates/. If detection is ambiguous, this refuses to
guess -- pass --type import|export explicitly instead:

    python main.py path/to/invoice.pdf --type import

Writes the full structured JSON (one key per template sheet) to disk, plus
a warnings list for anything dropped or unparseable, and prints a summary.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from schema import load_template_schema
from pdf_reader import extract_text, check_scan_quality, ClippedTableError
from reextraction import extract_and_validate, MAX_REEXTRACTION_ATTEMPTS
from detect_type import detect_shipment_type, AmbiguousShipmentTypeError
from templates_config import TEMPLATE_PATH, ITEM_SHEET_NAME

# (serial, description, qty, unit, unit_price, line_total-or-None) -- the
# two templates don't share column names, so the human-readable summary
# needs to know which columns to read for whichever type this run is.
_SUMMARY_FIELDS = {
    "export": ("Item_Ser_No", "Item_Desc", "Item_Qty", "Item_Unit1", "Item_Unit_Price", "ItmTaxableVal"),
    "import": ("SL_No", "Item_Desc1", "Item_Qty", "Item_Unit", "Item_Unit_Price", None),
}


def _get_val(x: Any) -> Any:
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    return x if x is not None else ""


async def run(
    pdf_path: str,
    model: str,
    out_path: str | None,
    shipment_type: str = "auto",
    template_path: str | None = None,
    force_ocr: bool | None = None,
    max_retries: int = MAX_REEXTRACTION_ATTEMPTS,
    allow_clipped: bool = False,
) -> dict:
    start_time = time.perf_counter()

    # A clipped table border means a row/column of the ACTUAL table is
    # physically missing from the scan -- not a quality problem OCR or the
    # LLM can work around. Checked here, BEFORE extract_text() runs (this
    # cheap check reuses the same scan_validator result extract_text()
    # would otherwise compute internally, but doesn't pay for the OCR pass
    # that follows it): an extraction run against known-incomplete source
    # data is a wasted OCR pass and LLM call producing a result nobody
    # should trust as complete. Confirmed real case: 24299.pdf's scan had
    # "Table border line(s) touch the page edge at: bottom, top" flagged
    # by this same check, yet a prior run let extraction proceed anyway --
    # spending 443s of OCR plus an LLM call -- before producing a
    # plausible-looking but silently incomplete item table; catching it
    # here instead means the file never reaches OCR or the LLM at all.
    # `allow_clipped=True` is the explicit override for a human who's
    # confirmed the clipping doesn't affect the fields they need, the same
    # role --type plays for an unresolved shipment-type detection below.
    precheck_scan_quality = check_scan_quality(pdf_path)
    if precheck_scan_quality and precheck_scan_quality.get("clipped") and not allow_clipped:
        raise ClippedTableError(
            f"Table border(s) touch the page edge ({', '.join(precheck_scan_quality['clipped_edges'])}) -- "
            "part of the table is likely missing from the scan. Re-scan the document, or "
            "re-run with allow_clipped=True (--allow-clipped on the CLI) to extract anyway."
        )

    text, tables_text, method, ocr_time_seconds, scan_quality = extract_text(pdf_path, force_ocr=force_ocr)
    if not text:
        raise RuntimeError(f"No text could be extracted from {pdf_path}, even with OCR.")
    print(f"[info] extracted {len(text)} chars from {pdf_path} via {method}")
    if ocr_time_seconds > 0:
        print(f"[info] OCR completed in {ocr_time_seconds:.2f}s using {method}")

    scan_warnings: list[str] = []
    if scan_quality and not scan_quality["valid"]:
        for reason in scan_quality["reasons"]:
            print(f"[warn] scan quality: {reason}")
            scan_warnings.append(f"scan quality: {reason}")

    detection = None
    if shipment_type == "auto":
        detection = detect_shipment_type(text)
        # A native digital-text extraction landing on "unknown" is sometimes
        # not genuinely ambiguous text -- it's a source PDF whose own
        # embedded text layer has a defect (confirmed real case: a country
        # name silently merged with a stray trailing character in the PDF's
        # own content stream, breaking the word-boundary match that would
        # otherwise have found it -- forcing this same file through OCR
        # re-reads the rendered pixels fresh and produces the clean word,
        # since OCR doesn't inherit whatever corrupted the source's text
        # objects). Only worth retrying when the FIRST pass used the fast
        # digital-text path (method == "2d_layout_canvas"): if it was
        # already OCR'd and still came back unknown, forcing OCR again
        # can't produce different text, so there's nothing to gain by
        # spending another OCR pass. This only adds cost on the failure
        # path that already has no usable result -- it can't change
        # anything for a file that already detects cleanly on its first
        # pass.
        if detection.shipment_type == "unknown" and method == "2d_layout_canvas":
            print("[info] shipment type unknown from digital text layer, retrying via forced OCR")
            ocr_text, ocr_tables_text, ocr_method, ocr_time_seconds2, ocr_scan_quality = extract_text(
                pdf_path, force_ocr=True,
            )
            ocr_detection = detect_shipment_type(ocr_text)
            if ocr_detection.shipment_type != "unknown":
                print(f"[info] forced-OCR retry resolved shipment type: {ocr_detection.shipment_type} (signal={ocr_detection.signal})")
                text, tables_text, method, scan_quality = ocr_text, ocr_tables_text, ocr_method, ocr_scan_quality
                ocr_time_seconds += ocr_time_seconds2
                detection = ocr_detection
        if detection.shipment_type == "unknown":
            raise AmbiguousShipmentTypeError(
                "Could not confidently determine whether this invoice is an "
                "import or export shipment (export_score="
                f"{detection.export_score}, import_score={detection.import_score}). "
                "Re-run with --type import or --type export."
            )
        resolved_type = detection.shipment_type
        print(f"[info] detected shipment type: {resolved_type} (signal={detection.signal})")
    else:
        resolved_type = shipment_type
        print(f"[info] shipment type set explicitly: {resolved_type}")

    template = Path(template_path) if template_path else TEMPLATE_PATH[resolved_type]
    schema = load_template_schema(template)

    extraction = await extract_and_validate(
        text, schema, resolved_type, model=model, tables_text=tables_text, max_retries=max_retries,
    )
    header = extraction["header"]
    sheets = extraction["sheets"]
    warnings = extraction["warnings"]
    token_usage = extraction["token_usage"]
    validation = extraction["validation"]

    if validation["valid"]:
        print(f"[info] validation passed on attempt {extraction['attempts']}/{max_retries + 1}")
    else:
        print(
            f"[warn] validation FAILED after {extraction['attempts']}/{max_retries + 1} attempt(s): "
            f"{len(validation['errors'])} unresolved error(s)"
        )
        for err in validation["errors"]:
            print(f"[warn]   [{err['rule']}] {err['message']}")

    extraction_time_seconds = round(time.perf_counter() - start_time, 2)
    item_sheet_name = ITEM_SHEET_NAME[resolved_type]
    line_items = sheets.get(item_sheet_name, [])

    result = {
        "source_file": Path(pdf_path).name,
        "shipment_type": resolved_type,
        "detection": (
            {
                "export_score": detection.export_score,
                "import_score": detection.import_score,
                "signal": detection.signal,
            }
            if detection else None
        ),
        "template": str(template),
        "extraction_method": method,
        "ocr_time_seconds": ocr_time_seconds,
        "scan_quality": scan_quality,
        "model": model,
        "extraction_time_seconds": extraction_time_seconds,
        "token_usage": token_usage,
        "header": header,
        "sheets": sheets,
        "warnings": scan_warnings + warnings,
        "validation": validation,
        "validation_history": extraction["validation_history"],
        "attempts": extraction["attempts"],
        "extraction_status": extraction["extraction_status"],
    }

    if out_path:
        # Valid and validation_failed results are written to separate
        # subfolders (spec section 13) rather than mixed together, so a
        # failed extraction never gets silently treated as a normal
        # successful one just because a file exists at the usual path --
        # and so difficult invoices needing a manual look are easy to
        # find later without re-reading every JSON's "valid" flag.
        out_p = Path(out_path)
        subdir = "valid" if validation["valid"] else "validation_failed"
        save_path = out_p.parent / subdir / out_p.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[info] wrote {save_path}")

    # Display Invoice Header Summary
    if header:
        inv_no = _get_val(header.get("invoice_number")) or "N/A"
        inv_date = _get_val(header.get("invoice_date")) or "N/A"
        supplier = _get_val(header.get("supplier_name")) or "N/A"
        buyer = _get_val(header.get("buyer_name")) or "N/A"
        total = _get_val(header.get("invoice_total")) or "N/A"
        curr = _get_val(header.get("currency")) or ""
        print(f"\n--- HEADER SUMMARY ---")
        print(f"Invoice No : {inv_no}  |  Date: {inv_date}")
        print(f"Supplier   : {supplier}")
        print(f"Buyer      : {buyer}")
        consignee = _get_val(header.get("consignee_name"))
        if consignee:
            print(f"Consignee  : {consignee}")
        print(f"Total      : {total} {curr}")
        terms = _get_val(header.get("payment_terms"))
        if terms:
            print(f"Terms      : {terms}")

    # Display Line Items Summary
    if line_items:
        ser_f, desc_f, qty_f, unit_f, price_f, total_f = _SUMMARY_FIELDS[resolved_type]
        print(f"\n--- {item_sheet_name} ({len(line_items)} items) ---")
        for itm in line_items:
            sno = _get_val(itm.get(ser_f)) or ""
            desc = str(_get_val(itm.get(desc_f)) or "")[:35]
            part = str(_get_val(itm.get("part_number")) or "")
            ord_no = str(_get_val(itm.get("order_number")) or "")
            qty = _get_val(itm.get(qty_f)) or ""
            uom = _get_val(itm.get(unit_f)) or ""
            price = _get_val(itm.get(price_f)) or ""
            ltot = _get_val(itm.get(total_f)) if total_f else ""
            print(f"  #{sno!s:<2} | {desc:<35} | Part: {part:<12} | Order: {ord_no:<12} | Qty: {qty} {uom} | Rate: {price} | Total: {ltot}")

    print(f"\n--- PERFORMANCE SUMMARY ---")
    print(f"Extraction Method        : {method}")
    if ocr_time_seconds > 0:
        print(f"OCR Execution Time       : {ocr_time_seconds:.2f}s")
    print(f"Total Extraction Time    : {extraction_time_seconds:.2f}s")

    if token_usage:
        print(f"\n--- TOKEN USAGE & COST ---")
        print(f"Input Tokens             : {token_usage.get('input_tokens', 0):,}")
        print(f"Cached Input Tokens      : {token_usage.get('cached_input_tokens', 0):,}")
        print(f"Output Tokens            : {token_usage.get('output_tokens', 0):,}")
        print(f"Total Tokens             : {token_usage.get('total_tokens', 0):,}")
        print(f"Cost (USD)               : ${token_usage.get('cost_usd', 0):.6f}")
        print(f"Cost (INR)               : INR {token_usage.get('cost_inr', 0):.4f}")

    if warnings:
        print(f"[warn] {len(warnings)} warning(s) -- see output file for details")

    return result


def main():
    ap = argparse.ArgumentParser(description="Extract an invoice PDF into the customs template schema.")
    ap.add_argument("pdf", help="Path to the invoice PDF")
    ap.add_argument("--type", choices=["import", "export", "auto"], default="auto",
                    help="Shipment type. 'auto' (default) detects it from the invoice text; "
                         "pass this explicitly if detection is ambiguous.")
    ap.add_argument("--template", default=None,
                    help="Override the template path (default: templates/EXP_TEMPLET.xlsx "
                         "or templates/IMP_TEMPLET.xlsx, chosen by --type)")
    ap.add_argument("--model", default="gpt-5-nano", help="OpenAI model to use")
    ap.add_argument("--out", default="poochi", help="Where to write the output JSON")
    ap.add_argument("--force-ocr", action="store_true", default=None,
                    help="Always skip the digital text-layer check and OCR every page "
                         "unconditionally (OCRmyPDF --force-ocr instead of --skip-text). "
                         "Without this flag, --skip-text vs --force-ocr is auto-decided per "
                         "file based on whether any page has a partial text layer -- pass "
                         "this only to override that and always force it.")
    ap.add_argument("--max-retries", type=int, default=MAX_REEXTRACTION_ATTEMPTS,
                    help=f"Re-extraction attempts allowed after a validation failure "
                         f"(default: {MAX_REEXTRACTION_ATTEMPTS}, i.e. {MAX_REEXTRACTION_ATTEMPTS + 1} "
                         f"attempts total). Set to 0 to disable retries.")
    ap.add_argument("--allow-clipped", action="store_true",
                    help="Extract anyway even if the scan's table border touches the page "
                         "edge (a row/column is likely physically missing from the scan). "
                         "Without this flag, extraction refuses to run on a confirmed-clipped "
                         "scan rather than spend an LLM call on data known to be incomplete.")
    args = ap.parse_args()

    out = args.out or (Path(args.pdf).stem + "_extracted.json")
    try:
        asyncio.run(run(
            args.pdf, args.model, out,
            shipment_type=args.type, template_path=args.template, force_ocr=args.force_ocr,
            max_retries=args.max_retries, allow_clipped=args.allow_clipped,
        ))
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
