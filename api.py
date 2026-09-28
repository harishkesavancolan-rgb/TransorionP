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

Both templates are parsed into schemas ONCE at startup (not per-request) --
restart the server if you edit either xlsx file in templates/.
"""
from __future__ import annotations
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

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
    text, tables_text, method, ocr_time_seconds, scan_quality = extract_text(tmp_path, force_ocr=force_ocr)
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

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append({"source_file": file.filename, "error": "not a PDF, skipped"})
            continue
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp.write(await file.read())
            tmp.close()  # release the handle so PDF readers can open it (required on Windows)
            results.append(await _process_pdf(tmp.name, file.filename, model, type, force_ocr=force_ocr, max_retries=max_retries))
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

    return JSONResponse({"results": results})
