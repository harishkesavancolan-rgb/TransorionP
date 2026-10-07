"""
api.py
-------
FastAPI wrapper around the extraction pipeline (schema.py, pdf_reader.py,
llm_extract.py, validate.py, detect_type.py -- unchanged).

Run:
    export OPENAI_API_KEY=sk-...
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health
         -> {"status": "ok", "templates": {"export": [...sheets], "import": [...sheets]}}

    POST /extract
         multipart/form-data, field name "file" = a single invoice PDF
         optional query params: ?model=gpt-5-nano  ?type=import|export  ?force_ocr=true
         ?max_retries=2 (re-extraction attempts after a failed validation;
         see invoice_validator.py / reextraction.py)
         (type omitted -> auto-detected from the invoice text; a 422 is
         returned if detection can't confidently tell import from export.
         force_ocr omitted -> --skip-text vs --force-ocr for the OCR pass is
         auto-decided per file based on whether any page has a partial text
         layer; force_ocr=true always skips the digital check and forces
         OCR on every page -- use it only to override the auto-decision)
         -> {"source_file", "shipment_type", "extraction_method", "model",
             "header", "sheets", "warnings", "validation",
             "validation_history", "attempts", "extraction_status", ...}

    POST /extract-batch
         multipart/form-data, field name "files" = one or more invoice PDFs
         optional query params: ?model=gpt-5-nano  ?type=import|export  ?force_ocr=true
         -> {"results": [ ...one result object per file... ]}

    POST /save-pdf?job_id=123&shipment_type=import|export
         multipart/form-data, field name "files" = one or more invoice PDFs
         Saves ALL the PDFs under received_pdfs/<job_id>/ (under their own names) first, then extracts from the
         saved copies (same pipeline as /extract-batch). Returns, per PDF:
         {"job_id", "file_name", "saved_path", "document_type": "selectable"|"scanned"|null,
          "extraction_status": "success"|"failed", "extracted_json", "error"}.
         "success" = the extraction ran and produced JSON (whether it passed validation
         is in extracted_json["extraction_status"]); "failed" = no JSON could be produced.
         When the extractions are done the results are also POSTed to a callback API
         (see _send_results_callback) and the outcome is returned under "callback".

Saved output: every successful extraction (either endpoint) is also written
to chandru_api/valid/<pdf name>.json or chandru_api/validation_failed/<pdf name>.json
next to this file (override the folder with EXTRACTOR_SAVE_DIR). Files that
errored before an extraction existed (not a PDF, no text, LLM failure) are not saved.

Both templates are parsed into schemas ONCE at startup (not per-request) --
restart the server if you edit either xlsx file in templates/.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from schema import load_template_schema
from pdf_reader import extract_text
from reextraction import extract_and_validate, MAX_REEXTRACTION_ATTEMPTS
from detect_type import detect_shipment_type
from templates_config import TEMPLATE_PATH

logger = logging.getLogger("invoice_extractor")
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = os.environ.get("EXTRACTOR_MODEL", "gpt-5-nano")

# Every successful /extract and /extract-batch result is also written here,
# split like main.py/batch.py do: valid/ and validation_failed/ subfolders.
SAVE_DIR = Path(os.environ.get("EXTRACTOR_SAVE_DIR", Path(__file__).resolve().parent / "chandru_api"))

# /save-pdf keeps the uploaded PDFs here, one subfolder per job_id, and
# extracts from the saved copy.
RECEIVED_PDF_DIR = Path(os.environ.get("EXTRACTOR_PDF_DIR", Path(__file__).resolve().parent / "received_pdfs"))

_schemas: dict[str, dict] = {}  # {"export": schema, "import": schema} -- loaded on startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    for shipment_type, path in TEMPLATE_PATH.items():
        if not path.exists():
            raise RuntimeError(f"Template not found at {path}")
        _schemas[shipment_type] = load_template_schema(path)
    yield


app = FastAPI(
    title="Invoice Extractor API",
    description="Extracts invoice PDFs into the customs export/import template schema.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "templates": {t: list(s.keys()) for t, s in _schemas.items()},
        "default_model": DEFAULT_MODEL,
    }


async def _process_pdf(
    tmp_path: str, filename: str, model: str, requested_type: str | None, force_ocr: bool | None = None,
    max_retries: int = MAX_REEXTRACTION_ATTEMPTS,
) -> dict:
    start_time = time.perf_counter()
    # Blocking (PDF parsing / OCR, up to minutes): run it in a worker thread so the
    # event loop stays free to receive other requests' uploads meanwhile.
    text, tables_text, method, ocr_time_seconds, scan_quality = await asyncio.to_thread(
        extract_text, tmp_path, force_ocr=force_ocr,
    )
    if not text:
        raise HTTPException(
            status_code=422,
            detail=f"No text could be extracted from '{filename}', even with OCR.",
        )

    scan_warnings: list[str] = []
    if scan_quality and not scan_quality["valid"]:
        scan_warnings = [f"scan quality: {reason}" for reason in scan_quality["reasons"]]

    detection = None
    if requested_type in ("import", "export"):
        resolved_type = requested_type
    else:
        detection = detect_shipment_type(text)
        if detection.shipment_type == "unknown":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Could not confidently determine whether '{filename}' is an import or "
                    f"export shipment (export_score={detection.export_score}, "
                    f"import_score={detection.import_score}). Resupply with ?type=import or ?type=export."
                ),
            )
        resolved_type = detection.shipment_type

    schema = _schemas[resolved_type]

    try:
        extraction = await extract_and_validate(
            text, schema, resolved_type, model=model, tables_text=tables_text, max_retries=max_retries,
        )
    except Exception as e:
        logger.exception(f"LLM extraction failed for '{filename}'")
        raise HTTPException(
            status_code=502,
            detail=f"LLM extraction failed for '{filename}': {e}",
        )

    extraction_time_seconds = round(time.perf_counter() - start_time, 2)

    return {
        "source_file": filename,
        "shipment_type": resolved_type,
        "detection": (
            {"export_score": detection.export_score, "import_score": detection.import_score, "signal": detection.signal}
            if detection else None
        ),
        "extraction_method": method,
        "ocr_time_seconds": ocr_time_seconds,
        "scan_quality": scan_quality,
        "model": model,
        "extraction_time_seconds": extraction_time_seconds,
        "token_usage": extraction["token_usage"],
        "header": extraction["header"],
        "sheets": extraction["sheets"],
        "warnings": scan_warnings + extraction["warnings"],
        "validation": extraction["validation"],
        "validation_history": extraction["validation_history"],
        "attempts": extraction["attempts"],
        "extraction_status": extraction["extraction_status"],
    }


def _save_result(result: dict, used: set[str] | None = None) -> None:
    """Write one extraction result to SAVE_DIR/<valid|validation_failed>/<pdf stem>.json.

    Never raises: a disk problem must not turn a finished (and paid-for)
    extraction into an error response. `used` holds the names already
    written in the current request, so two uploads with the same filename in
    one batch get "_2", "_3"... instead of overwriting each other; a later
    request with the same filename replaces the earlier file, as batch.py does.
    """
    try:
        stem = Path(str(result.get("source_file") or "unnamed")).stem or "unnamed"  # .name/.stem drops any path parts
        subdir = "valid" if result.get("validation", {}).get("valid") else "validation_failed"
        name, n = stem, 1
        while used is not None and f"{subdir}/{name}".lower() in used:
            n += 1
            name = f"{stem}_{n}"
        if used is not None:
            used.add(f"{subdir}/{name}".lower())
        path = SAVE_DIR / subdir / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved extraction -> %s", path)
    except Exception:
        logger.exception("Could not save extraction for %r to %s", result.get("source_file"), SAVE_DIR)


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    model: str = Query(default=None),
    type: str | None = Query(default=None, pattern="^(import|export)$"),
    force_ocr: bool | None = Query(default=None),
    max_retries: int = Query(default=MAX_REEXTRACTION_ATTEMPTS, ge=0,
                              description="Re-extraction attempts allowed after a validation failure."),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    model = model or DEFAULT_MODEL

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(await file.read())
        tmp.close()  # release the handle so PDF readers can open it (required on Windows)
        result = await _process_pdf(tmp.name, file.filename, model, type, force_ocr=force_ocr, max_retries=max_retries)
        _save_result(result)
    finally:
        try:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
        except Exception:
            pass

    return JSONResponse(result)


@app.post("/extract-batch")
async def extract_batch(
    files: list[UploadFile] = File(...),
    model: str = Query(default=None),
    type: str | None = Query(default=None, pattern="^(import|export)$"),
    force_ocr: bool | None = Query(default=None),
    max_retries: int = Query(default=MAX_REEXTRACTION_ATTEMPTS, ge=0,
                              description="Re-extraction attempts allowed after a validation failure."),
):
    model = model or DEFAULT_MODEL
    results = []
    used_names: set[str] = set()

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append({"source_file": file.filename, "error": "not a PDF, skipped"})
            continue
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp.write(await file.read())
            tmp.close()  # release the handle so PDF readers can open it (required on Windows)
            result = await _process_pdf(tmp.name, file.filename, model, type, force_ocr=force_ocr, max_retries=max_retries)
            _save_result(result, used_names)
            results.append(result)
        except HTTPException as e:
            results.append({"source_file": file.filename, "error": e.detail})
        except Exception as e:
            results.append({"source_file": file.filename, "error": str(e)})
        finally:
            try:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
            except Exception:
                pass
        # print({"results": results})
    return JSONResponse({"results": results})


def _document_type(extraction_method: str | None) -> str | None:
    """"selectable" if the text came straight from the PDF's own text layer,
    "scanned" if it needed OCR (ocrmypdf_docker / ocr_<engine>); None if unknown."""
    if not extraction_method:
        return None
    return "selectable" if extraction_method == "2d_layout_canvas" else "scanned"


def _callback_config() -> tuple[str, dict[str, str]]:
    """(url, headers) for the results callback, read from the environment on each
    call. Credentials are never kept in the source:
        EXTRACTOR_CALLBACK_URL           default http://192.168.2.46:8765/process_invoices/
        EXTRACTOR_CALLBACK_API_KEY       -> X-API-KEY header
        EXTRACTOR_CALLBACK_BEARER_TOKEN  -> Authorization: Bearer <token> (usually short-lived)
    Set EXTRACTOR_CALLBACK_URL to an empty string to switch the callback off."""
    url = os.environ.get("EXTRACTOR_CALLBACK_URL", "http://192.168.2.46:8765/process_invoices/")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("EXTRACTOR_CALLBACK_API_KEY"):
        headers["X-API-KEY"] = os.environ["EXTRACTOR_CALLBACK_API_KEY"]
    if os.environ.get("EXTRACTOR_CALLBACK_BEARER_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["EXTRACTOR_CALLBACK_BEARER_TOKEN"]
    return url, headers


async def _send_results_callback(job_id: int, results: list[dict]) -> dict:
    """POST {"job_id", "results"} to the callback API. Never raises -- a callback
    problem must not turn finished extractions into an error response; the
    outcome is returned so /save-pdf can report it."""
    url, headers = _callback_config()
    if not url:
        return {"sent": False, "reason": "callback disabled (EXTRACTOR_CALLBACK_URL is empty)"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, content=json.dumps(
                {"job_id": job_id, "results": results}, ensure_ascii=False).encode("utf-8"))
        ok = resp.is_success
        if not ok:
            logger.warning("Results callback for job %s got HTTP %s: %s", job_id, resp.status_code, resp.text[:300])
        else:
            logger.info("Results callback for job %s -> HTTP %s", job_id, resp.status_code)
        return {"sent": ok, "status_code": resp.status_code, **({} if ok else {"response": resp.text[:300]})}
    except Exception as e:
        logger.exception("Results callback for job %s failed", job_id)
        return {"sent": False, "error": str(e)}


@app.post("/save-pdf")
async def save_pdf(
    files: list[UploadFile] = File(...),
    job_id: int = Query(...),
    shipment_type: str = Query(..., pattern="^(import|export)$"),
):
    """Save the uploaded PDFs under received_pdfs/<job_id>/ and extract them
    from the saved copies. One result per PDF, each carrying the job_id."""
    job_folder = RECEIVED_PDF_DIR / str(job_id)
    job_folder.mkdir(parents=True, exist_ok=True)

    # PHASE 1 -- save EVERY uploaded PDF before any extraction starts, so all
    # files are on disk (and their saved_path is known) first.
    entries: list[dict] = []
    used_names: set[str] = set()
    for file in files:
        if not file.filename:
            continue
        original_filename = Path(file.filename).name  # drops any path parts
        entry = {
            "job_id": job_id,
            "file_name": original_filename,
            "saved_path": None,
            "document_type": None,
            "extraction_status": "failed",
            "extracted_json": None,
            "error": None,
        }
        entries.append(entry)
        try:
            if not original_filename.lower().endswith(".pdf"):
                entry["error"] = "Only PDF files are accepted."
                continue
            # Saved under its own name. Two files with the SAME name in this one
            # request would otherwise overwrite each other (and both entries
            # would then extract the same file), so the later one gets "_2", "_3"...
            stem, suffix = Path(original_filename).stem, Path(original_filename).suffix
            n, name = 1, original_filename
            while name.lower() in used_names:
                n += 1
                name = f"{stem}_{n}{suffix}"
            used_names.add(name.lower())
            pdf_path = job_folder / name
            pdf_path.write_bytes(await file.read())
            entry["saved_path"] = str(pdf_path)
        except Exception as e:
            logger.exception("save-pdf could not save %r", original_filename)
            entry["error"] = f"Could not save the file: {e}"
        finally:
            await file.close()

    # PHASE 2 -- extract from the saved copies, one after another.
    for entry in entries:
        if not entry["saved_path"]:
            continue
        try:
            extraction = await _process_pdf(
                entry["saved_path"], entry["file_name"], DEFAULT_MODEL, shipment_type,
                force_ocr=None, max_retries=MAX_REEXTRACTION_ATTEMPTS,
            )
            entry["extracted_json"] = extraction
            entry["document_type"] = _document_type(extraction.get("extraction_method"))
            entry["extraction_status"] = "success"
        except HTTPException as e:
            entry["error"] = str(e.detail)
        except Exception as e:
            logger.exception("save-pdf extraction failed for %r", entry["file_name"])
            entry["error"] = str(e)

    # Hand the finished extractions to the callback API in the /extract-batch
    # shape: the extraction result per file, or {"source_file", "error"} for a
    # file that produced none.
    # "document_type" (selectable / scanned) is added to each result.
    callback_results = [
        {**e["extracted_json"], "document_type": e["document_type"]} if e["extracted_json"] is not None
        else {"source_file": e["file_name"], "document_type": None, "error": e["error"]}
        for e in entries
    ]
    callback = await _send_results_callback(job_id, callback_results)

    return JSONResponse({"job_id": job_id, "shipment_type": shipment_type, "results": entries, "callback": callback})
