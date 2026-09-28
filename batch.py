"""
batch.py
---------
Run extraction over every PDF in a folder (recursively). Writes one JSON
per invoice plus a combined summary CSV so you can eyeball which invoices
had warnings, came back empty, or need a manual --type override, without
opening all of them.

Usage:
    export OPENAI_API_KEY=sk-...
    python batch.py path/to/invoices_root --out-dir results/

    # Run 8 files at a time (faster, uses more API quota):
    python batch.py path/to/invoices_root --out-dir results/ --workers 8

Each file's shipment type is auto-detected independently (a folder can
freely mix import and export invoices). Pass --type import|export to force
every file in the batch to one type instead (skips per-file detection).
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

from main import run
from detect_type import AmbiguousShipmentTypeError
from reextraction import MAX_REEXTRACTION_ATTEMPTS

_EMPTY_ROW_FIELDS = {
    "method": "", "ocr_time_sec": 0.0, "time_sec": 0.0,
    "item_rows": 0, "total_tokens": 0, "cost_inr": 0.0, "warnings": 0,
    "extraction_status": "", "attempts": 0, "validation_errors": 0,
}


async def _process_one(
    pdf: Path,
    model: str,
    out_dir: Path,
    shipment_type: str,
    force_ocr: bool | None,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
) -> dict:
    """Process a single PDF, bounded by the semaphore for concurrency control."""
    async with semaphore:
        out_path = out_dir / f"{pdf.stem}.json"
        print(f"[{idx}/{total}] starting  {pdf.name}")
        t0 = time.perf_counter()
        try:
            result = await run(
                str(pdf), model, str(out_path), shipment_type=shipment_type,
                force_ocr=force_ocr, max_retries=max_retries,
            )
            elapsed = round(time.perf_counter() - t0, 1)
            token_info = result.get("token_usage", {})
            n_items = sum(len(rows) for rows in result.get("sheets", {}).values())
            n_warn = len(result.get("warnings", []))
            print(f"[{idx}/{total}] done      {pdf.name}  "
                  f"({n_items} items, {n_warn} warnings, {elapsed}s)")
            return {
                "file": pdf.name,
                "status": "ok",
                "shipment_type": result.get("shipment_type", ""),
                "method": result.get("extraction_method", ""),
                "ocr_time_sec": result.get("ocr_time_seconds", 0.0),
                "time_sec": elapsed,
                "item_rows": n_items,
                "total_tokens": token_info.get("total_tokens", 0),
                "cost_inr": token_info.get("cost_inr", 0.0),
                "warnings": n_warn,
                "extraction_status": result.get("extraction_status", ""),
                "attempts": result.get("attempts", 1),
                "validation_errors": len(result.get("validation", {}).get("errors", [])),
            }
        except AmbiguousShipmentTypeError as e:
            print(f"[{idx}/{total}] skipped   {pdf.name}: ambiguous type -- re-run with --type export|import")
            return {
                "file": pdf.name,
                "status": "skipped: unknown shipment type -- needs --type override",
                "shipment_type": "unknown",
                **_EMPTY_ROW_FIELDS,
            }
        except Exception as e:
            print(f"[{idx}/{total}] FAILED    {pdf.name}: {e}")
            return {
                "file": pdf.name,
                "status": f"FAILED: {e}",
                "shipment_type": "",
                **_EMPTY_ROW_FIELDS,
            }


async def run_batch(
    folder: str,
    model: str,
    out_dir_path: str,
    shipment_type: str = "auto",
    force_ocr: bool | None = None,
    max_retries: int = MAX_REEXTRACTION_ATTEMPTS,
    workers: int = 5,
):
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(Path(folder).rglob("*.pdf"))
    total = len(pdfs)
    print(f"[info] found {total} PDF(s) -- running {workers} concurrently")

    semaphore = asyncio.Semaphore(workers)
    t_start = time.perf_counter()

    tasks = [
        _process_one(pdf, model, out_dir, shipment_type, force_ocr, max_retries, semaphore, i + 1, total)
        for i, pdf in enumerate(pdfs)
    ]
    summary_rows = await asyncio.gather(*tasks)

    elapsed_total = round(time.perf_counter() - t_start, 1)
    ok = sum(1 for r in summary_rows if r["status"] == "ok")
    failed = sum(1 for r in summary_rows if r["status"].startswith("FAILED"))
    skipped = total - ok - failed
    total_cost = round(sum(r["cost_inr"] for r in summary_rows), 2)

    print(f"\n[done] {ok}/{total} succeeded, {failed} failed, {skipped} skipped "
          f"in {elapsed_total}s -- total cost Rs.{total_cost}")

    summary_path = out_dir / "_summary.csv"
    fieldnames = [
        "file", "status", "shipment_type", "method", "ocr_time_sec",
        "time_sec", "item_rows", "total_tokens", "cost_inr", "warnings",
        "extraction_status", "attempts", "validation_errors",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[info] wrote summary -> {summary_path}")


def main():
    # Windows consoles default to a legacy codepage (cp1252) that can't
    # encode every character batch.py might print (invoice text/names
    # extracted from PDFs is arbitrary Unicode) -- reconfigure stdout/stderr
    # to UTF-8 so a single odd character can't crash the whole batch run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Root folder containing invoice PDFs (searched recursively)")
    ap.add_argument("--type", choices=["import", "export", "auto"], default="auto",
                    help="Force every file to this shipment type instead of auto-detecting per file")
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--out-dir", default="result_fused1")
    ap.add_argument("--workers", type=int, default=5,
                    help="How many PDFs to process concurrently (default: 5). "
                         "Increase if your API tier allows higher rate limits, "
                         "decrease if you hit rate-limit errors.")
    ap.add_argument("--force-ocr", action="store_true", default=None,
                    help="Apply to every file: always skip the digital text-layer check and "
                         "OCR unconditionally (OCRmyPDF --force-ocr instead of --skip-text). "
                         "Without this flag, --skip-text vs --force-ocr is auto-decided per "
                         "file -- pass this only to override that and always force it.")
    ap.add_argument("--max-retries", type=int, default=MAX_REEXTRACTION_ATTEMPTS,
                    help=f"Re-extraction attempts allowed per file after a validation failure "
                         f"(default: {MAX_REEXTRACTION_ATTEMPTS}, i.e. {MAX_REEXTRACTION_ATTEMPTS + 1} "
                         f"attempts total). Set to 0 to disable retries.")
    args = ap.parse_args()

    asyncio.run(run_batch(
        args.folder, args.model, args.out_dir, shipment_type=args.type,
        force_ocr=args.force_ocr, max_retries=args.max_retries, workers=args.workers,
    ))


if __name__ == "__main__":
    main()
