# Invoice → Customs Template Extractor

Extracts data from an invoice PDF into either the export template
(`templates/EXP_TEMPLET.xlsx`, 13 sheets, line items on `ITEM`) or the
import template (`templates/IMP_TEMPLET.xlsx`, 8 sheets, line items on
`BOE`), auto-detecting which one applies from the invoice text itself.

## How it works

1. **`pdf_reader.py`** extracts text via a 2D layout-preserving canvas
   built from `pdfplumber` word coordinates (keeps columns/tables intact).
   Genuine table-column boundaries get an explicit `│` marker instead of
   plain whitespace -- detected either from a gap recurring at the same
   x-position across multiple lines (the actual signature of a table
   column, vs. one line just having a wide gap after a label), or from a
   ruled line/rect the PDF itself drew as a column divider. Both prompts
   in `llm_extract.py` are told what `│` means, so the model doesn't
   merge text across a real column boundary into one field just because
   whitespace alone can't distinguish "two columns" from "one sentence
   with extra spacing." Whether a PDF gets this fast digital path at all
   is decided by
   `is_scanned_pdf()`, which checks text density **per page** (chars/page,
   words/page, single-character-word noise, how many pages are actually
   dense) -- not just "does the whole document have 40+ characters
   somewhere," which a single letterhead or stamped line on an otherwise
   all-image multi-page scan could pass on its own. If it's scanned (or
   `--force-ocr`/`force_ocr=True` is passed), it falls back through
   OCRmyPDF (Docker) → RapidOCR → PaddleOCR → PaddleOCR-VL, logging which
   tier ran and why any earlier tier was skipped. The Docker pass needs to
   pick one of two OCRmyPDF flags: `--skip-text` (fast, skips OCR on any
   page that already has *some* text layer, even a sliver of garbage) or
   `--force-ocr` (slower, re-OCRs every page unconditionally). By default
   this is auto-decided per file: if any page has a partial text layer
   (some real words, but fewer than a dense page would have -- exactly the
   band `--skip-text` mishandles, since it only checks "is there *any*
   text," not "is there *enough*"), `--force-ocr` is used; otherwise
   `--skip-text` is used since there's nothing on any page for it to
   wrongly skip. Pass `--force-ocr`/`force_ocr=True` (or `force_ocr=False`)
   to override the auto-decision either way. The Docker pass also now
   passes `--deskew` (fixes fine-grained scan tilt, on top of
   `--rotate-pages`'s coarse 90-degree-multiple correction).

   **Tilt on selectable PDFs** (real embedded text, not a scan) is handled
   differently: `scan_validator.py` renders each page and measures its
   tilt via a Hough-transform line search (works without any ruled
   table -- ordinary text baselines provide enough line segments;
   validated against synthetic rotations from 0-25 degrees, accurate to
   within a 1-degree bucket the whole way). For a correctable tilt, that
   angle is used to rotate each word's *center point* in coordinate space
   before line-grouping -- no re-rendering or re-OCR, which is the point:
   it's cheap enough to run on every page. (Center-point rotation, not a
   rotated bounding box: rotating a wide-but-short word's four corners
   and re-deriving an axis-aligned box from them measurably overshoots --
   confirmed empirically, 11-21pt of spurious vertical drift at just
   8-15 degrees, comfortably enough to re-break the very line-grouping it
   was meant to fix. Center-point rotation has no such artifact.)
   `scan_validator.validate_pdf()`'s result also flags the two cases that
   genuinely can't be fixed this way -- tilt beyond ~30 degrees, or a
   table border running off the page edge (content is physically missing,
   not just misaligned) -- surfaced as a `scan_quality` field and a
   warning in the output JSON rather than silently producing a bad
   extraction.
2. **`detect_type.py`** classifies the extracted text as an **import** or
   **export** shipment, using weighted keyword scoring (customs phrases
   like "Bill of Entry" / "Shipping Bill") plus structural corroboration
   (country of origin/destination relative to India). This runs before any
   LLM call and costs nothing extra. If it can't confidently decide, it
   says so (`"unknown"`) rather than guessing -- you're expected to pass
   `--type import` / `--type export` explicitly in that case.
3. **`schema.py`** reads the column headers straight from whichever
   template applies. The headers themselves are read dynamically, but a
   set of column-name-specific rules (aliases, forced string types,
   customs cleanup in `validate.py`) are intentionally hardcoded India-
   customs domain knowledge -- see "What's genuinely hardcoded" below.
4. **`llm_extract.py`** sends the invoice text to GPT (`gpt-5-nano` by
   default) in two concurrent passes -- one for header/document metadata,
   one for line items -- and returns generic, template-agnostic JSON. It
   knows nothing about `ITEM` vs. `BOE` column names. Network calls retry
   with backoff on transient errors (rate limits, timeouts, 5xx); it never
   retries auth or bad-request errors.
5. **`validate.py`** maps those generic results onto the chosen template's
   real columns (`_map_export_item` / `_map_import_item`), applies
   India-customs cleanup specific to that sheet, and coerces
   currency-formatted numbers ("$0.0510", "1,234") into real numbers.
   **Every template column not actually extracted is `""`** -- there is no
   guessed default (no auto-`"N"` for Y/N flags, no auto-`"P"` for
   `Printed`) and no confidence score anywhere in the output.
6. **`main.py`** / **`batch.py`** tie it together for one file or a whole
   folder tree; **`api.py`** exposes the same pipeline over HTTP.

## Setup

```bash
pip install -r requirements.txt

# OCR fallback also needs these system packages (only used if a PDF has
# no text layer at all — most invoices won't need it):
#   Ubuntu/Debian: sudo apt install tesseract-ocr poppler-utils
#   Mac:           brew install tesseract poppler

cp .env.example .env   # then fill in OPENAI_API_KEY
```

## Usage

**Single file (shipment type auto-detected):**
```bash
python main.py invoice.pdf --out result.json
```

**Single file (type known in advance, or detection was ambiguous):**
```bash
python main.py invoice.pdf --type import --out result.json
```

**Whole folder, recursive (mixed import/export invoices are fine — each
file is detected independently unless you force `--type`):**
```bash
python batch.py /path/to/invoices_root --out-dir results/
```
Writes one JSON per invoice into `results/`, plus `results/_summary.csv`
listing shipment type, item-row counts, and warning counts per file --
including any file skipped for an ambiguous type, so you know which ones
need a manual `--type` re-run.

**API:**
```bash
uvicorn api:app --reload --port 8000
# POST /extract        (multipart file, optional ?type=import|export)
# POST /extract-batch   (multipart files[])
# GET  /health
```

## Output shape

```json
{
  "source_file": "INV5.pdf",
  "shipment_type": "export",
  "detection": {"export_score": 5, "import_score": 0, "signal": "keyword"},
  "extraction_method": "2d_layout_canvas",
  "scan_quality": {"skew_deg": 1.0, "skew_ok": true, "clipped": false, "clipped_edges": [], "valid": true, "reasons": []},
  "model": "gpt-5-nano",
  "token_usage": {"total_tokens": 1834, "cost_usd": 0.000221, "cost_inr": 0.0188},
  "header": {"invoice_number": "INV5", "supplier_name": "...", "...": "..."},
  "sheets": {
    "ITEM": [
      {"Item_Ser_No": 1, "Item_Desc": "...", "Item_Qty": 2250, "Item_Unit_Price": 0.051, "...": "", "part_number": "...", "model_or_type": "...", "order_number": "..."}
    ],
    "DRAWBACK": [],
    "License": [],
    "...": []
  },
  "warnings": []
}
```

`sheets` always has one key per sheet in whichever template applied
(`ITEM` for export, `BOE` for import). Every row in the populated sheet
carries **every** column the template defines, in template order --
anything not actually found on the invoice is `""`. Two extra
non-template convenience fields (`part_number`, `model_or_type`) plus
`order_number` ride along on every row regardless of shipment type, since
they're useful commercial fields even when the template itself doesn't
have a dedicated column for them (the import `BOE` sheet's `Model` column
is also populated from the same value).

## Important, honest caveats

- **Most non-line-item sheets will legitimately come back empty most of
  the time.** On the export side: DRAWBACK, License, AR4, JOBWORK,
  THIRDPARTY, Constituent, Production, Control, ReExport, STR, InfoType.
  On the import side: most of `BOE`'s ~85 columns are BCD/CVD/SWC/IGST
  notification numbers, SIMS registration fields, and ADIC references.
  All of this is customs-*filing*-stage data (scheme codes, license
  registrations, notification numbers) that usually isn't printed on a
  commercial invoice at all -- it's added later by whoever files the
  shipping bill or bill of entry. Empty is the *correct* answer for most
  invoices here, not a bug.
- **The prompt forbids inventing values.** If a field isn't clearly in the
  invoice text, the model is told to omit it rather than guess -- this
  feeds a customs filing, where a wrong value is worse than a missing one.
  Always spot-check `warnings` and a sample of outputs against source PDFs
  before trusting a large batch.
- **Import/export detection can come back `"unknown"`.** This is
  deliberate -- the two templates are structurally incompatible (see
  below), so guessing wrong would silently misplace every line item. An
  ambiguous file needs a human to say `--type import` or `--type export`.
- Every column's data type (string vs. number) is *inferred from its
  name* (e.g. anything with "qty", "price", "amount" in it → number). If a
  field gets misclassified, adjust the hint lists at the top of
  `schema.py`.

## What's genuinely hardcoded (and has to stay that way)

- **Export and import templates have disjoint column sets.** The export
  template's line items live on `ITEM` (state/district code, FTA code,
  taxable value, ...); the import template's live on `BOE` (COO
  certificate, brand, model, dozens of notification-number pairs, ...) --
  there is no shared shape. `validate.py` has two separate mapping
  functions (`_map_export_item` / `_map_import_item`) because a single
  unified mapping genuinely isn't possible.
- **`schema.py`'s `_MANUAL_ALIASES` and `_STRING_FORCE`, and
  `validate.py`'s domain-cleanup rules** (telling a container number apart
  from an airway bill number, an AD Code apart from an FTA Code, HSN
  tariff-description trimming, state-code extraction from `"MH(12)"`-style
  strings, ...) are hand-curated India-customs knowledge tied to specific
  column names. If you rename a column in either xlsx template, the schema
  will still pick up the new header automatically -- but any rule written
  against the *old* name silently stops applying to it. There's no
  automatic detection of that; re-check these lists after renaming a
  column.

## Dev-only tools

`tools/benchmark_ocr.py` compares pdfplumber's native text layer vs.
RapidOCR vs. Docker OCRmyPDF timing/output on a given PDF. It's not
imported by `main.py`/`batch.py`/`api.py` -- safe to ignore or delete.
