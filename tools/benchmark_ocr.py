"""
benchmark_ocr.py
----------------
Dev-only utility, NOT part of the runtime extraction pipeline (main.py /
batch.py / api.py never import this). Safe to delete without affecting the
product; kept around for whoever is tuning OCR settings later.

CLI evaluation tool to compare and benchmark OCR engines on invoice PDFs:
1. Native PDFplumber (Digital Text Layer)
2. RapidOCR (ONNX Runtime / PP-OCRv4)
3. Docker OCRmyPDF (Tesseract LSTM)

Usage:
    python tools/benchmark_ocr.py path/to/invoice.pdf
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Lets `from pdf_reader import ...` resolve to the project root even though
# this script lives one directory down, in tools/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import tempfile
import subprocess
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from pdf_reader import extract_page_layout_canvas

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from rapidocr_onnxruntime import RapidOCR
    rapid_engine = RapidOCR()
    has_rapid = True
except Exception:
    rapid_engine = None
    has_rapid = False


def evaluate_pdf(pdf_path: str, dpi: int = 200, page_idx: int = 0):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"[error] PDF not found: {pdf_file}")
        return

    scale = dpi / 72
    doc = pdfium.PdfDocument(str(pdf_file))
    num_pages = len(doc)
    page = doc[page_idx]
    bitmap = page.render(scale=scale)
    page_img = bitmap.to_pil()

    print("=" * 80)
    print(f"[OCR BENCHMARK EVALUATION] {pdf_file.name} (Page {page_idx + 1} of {num_pages})")
    print("=" * 80)

    # 1. Native PDFplumber
    t0 = time.perf_counter()
    with pdfplumber.open(str(pdf_file)) as pdf:
        p_page = pdf.pages[page_idx]
        words = p_page.extract_words()
        plumb_text = extract_page_layout_canvas(p_page)
    plumb_time = round(time.perf_counter() - t0, 3)

    print(f"\n[1] Native PDFplumber Text Layer:")
    print(f"    - Latency       : {plumb_time:.2f}s")
    print(f"    - Words Detected: {len(words)}")
    print(f"    - Chars Detected: {len(plumb_text)}")
    print(f"    - Status        : {'[SCANNED / IMAGE ONLY] (0 text)' if len(plumb_text.strip()) < 40 else '[OK] Embedded digital text layer found'}")

    # 2. RapidOCR ONNX
    if has_rapid:
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
            page_img.save(tmp_f, format="PNG")
            tmp_f_path = tmp_f.name

        with open(tmp_f_path, "rb") as f:
            img_bytes = f.read()
        res, _ = rapid_engine(img_bytes)
        try:
            os.unlink(tmp_f_path)
        except Exception:
            pass
        rapid_time = round(time.perf_counter() - t0, 3)

        rapid_lines = [item[1] for item in res] if res else []
        rapid_confs = [float(item[2]) for item in res] if res else []
        rapid_words = sum(len(l.split()) for l in rapid_lines)
        avg_conf = (sum(rapid_confs) / len(rapid_confs)) if rapid_confs else 0.0

        print(f"\n[2] RapidOCR (ONNX Runtime / PP-OCRv4 CPU):")
        print(f"    - Latency       : {rapid_time:.2f}s")
        print(f"    - Lines / Boxes : {len(rapid_lines)}")
        print(f"    - Words Detected: {rapid_words}")
        print(f"    - Chars Detected: {sum(len(l) for l in rapid_lines)}")
        print(f"    - Avg OCR Conf  : {avg_conf * 100:.1f}%")
        print(f"    - Status        : [OK] Completed purely in Python (0 Docker / 0 C++ required)")

    # 3. Docker OCRmyPDF
    print(f"\n[3] Docker OCRmyPDF (Tesseract LSTM):")
    tmp_out = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_out.close()
    cmd = [
        "docker", "run", "--rm", "-i",
        "jbarlow83/ocrmypdf-alpine",
        "--image-dpi", str(dpi),
        "--skip-text",
        "-", "-"
    ]
    t0 = time.perf_counter()
    try:
        with open(pdf_file, "rb") as infile, open(tmp_out.name, "wb") as outfile:
            proc = subprocess.run(cmd, stdin=infile, stdout=outfile, stderr=subprocess.PIPE, timeout=90)
        docker_time = round(time.perf_counter() - t0, 3)
        if proc.returncode == 0 and os.path.getsize(tmp_out.name) > 0:
            with pdfplumber.open(tmp_out.name) as d_pdf:
                d_page = d_pdf.pages[page_idx]
                d_words = d_page.extract_words()
                d_text = extract_page_layout_canvas(d_page)
            print(f"    - Latency       : {docker_time:.2f}s")
            print(f"    - Words Detected: {len(d_words)}")
            print(f"    - Chars Detected: {len(d_text)}")
            print(f"    - Status        : [OK] Searchable PDF generated and text coordinates extracted")
        else:
            print(f"    - Status        : [WARN] Docker returned code {proc.returncode} (Docker Desktop not running or image missing)")
    except Exception as e:
        print(f"    - Status        : [WARN] Docker skipped: {e}")
    finally:
        if os.path.exists(tmp_out.name):
            try:
                os.unlink(tmp_out.name)
            except Exception:
                pass

    print("\n" + "=" * 80)
    print("  To inspect bounding boxes and text layout visually side-by-side:")
    print("  Open the Jupyter Notebook: invoice_extractor/ocr_benchmark.ipynb")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and benchmark OCR engines on invoice PDFs.")
    parser.add_argument("pdf", help="Path to invoice PDF")
    parser.add_argument("--page", type=int, default=0, help="Page index to evaluate (0-based, default: 0)")
    parser.add_argument("--dpi", type=int, default=200, help="Rendering DPI (default: 200)")
    args = parser.parse_args()

    evaluate_pdf(args.pdf, dpi=args.dpi, page_idx=args.page)


if __name__ == "__main__":
    main()
