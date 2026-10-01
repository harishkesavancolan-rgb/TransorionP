"""
Tests for batch.py's folder-level bookkeeping: output naming when the same
filename appears in more than one subfolder, and the summary CSV. main.run
(the per-file pipeline) is mocked, so no PDF is actually read and no LLM is
called -- the PDFs here are empty placeholder files.

Uses asyncio.run() directly rather than pytest-asyncio, same as
test_reextraction.py.
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import batch


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_unique_stems_keep_their_plain_name(tmp_path):
    a = _touch(tmp_path / "BECKMAN" / "24299.pdf")
    b = _touch(tmp_path / "GLASTRON" / "0715000217.pdf")
    assert batch._output_names([a, b], tmp_path) == {a: "24299", b: "0715000217"}


def test_shared_stems_get_their_folder_path_prefixed(tmp_path):
    a = _touch(tmp_path / "BECKMAN" / "New folder (2)" / "INPL.pdf")
    b = _touch(tmp_path / "GLASTRON" / "INPL.pdf")
    c = _touch(tmp_path / "GLASTRON" / "INPL2.pdf")
    names = batch._output_names([a, b, c], tmp_path)
    assert names[a] == "BECKMAN__New folder (2)__INPL"
    assert names[b] == "GLASTRON__INPL"
    assert names[c] == "INPL2"  # not shared -- left alone


def test_shared_stems_are_detected_case_insensitively(tmp_path):
    a = _touch(tmp_path / "A" / "Invoice.pdf")
    b = _touch(tmp_path / "B" / "INVOICE.pdf")
    names = batch._output_names([a, b], tmp_path)
    assert names[a] != names[b]
    assert names[a].lower() != names[b].lower()


def _fake_run(seen_out_paths):
    async def fake_run(pdf_path, model, out_path, **kwargs):
        seen_out_paths.append(Path(out_path).name)
        return {"shipment_type": "export", "extraction_method": "2d_layout_canvas",
                "sheets": {}, "warnings": [], "token_usage": {}, "validation": {}}
    return fake_run


def test_run_batch_writes_one_output_per_pdf_even_with_duplicate_names(tmp_path, monkeypatch):
    root = tmp_path / "in"
    _touch(root / "BECKMAN" / "INPL.pdf")
    _touch(root / "GLASTRON" / "INPL.pdf")
    seen: list[str] = []
    monkeypatch.setattr(batch, "run", _fake_run(seen))

    asyncio.run(batch.run_batch(str(root), "gpt-5-nano", str(tmp_path / "out")))

    assert sorted(seen) == ["BECKMAN__INPL.json", "GLASTRON__INPL.json"]


def test_summary_csv_handles_non_cp1252_filenames(tmp_path, monkeypatch):
    # Confirmed real filename in this project's sample data. The summary
    # used to be written with the platform default encoding (cp1252 on
    # Windows), which can't encode it -- crashing after every file had
    # already been processed.
    root = tmp_path / "in"
    _touch(root / "Mix (18)🏳️.pdf")
    monkeypatch.setattr(batch, "run", _fake_run([]))

    out_dir = tmp_path / "out"
    asyncio.run(batch.run_batch(str(root), "gpt-5-nano", str(out_dir)))

    with open(out_dir / "_summary.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["file"] for r in rows] == ["Mix (18)🏳️.pdf"]
    assert rows[0]["status"] == "ok"
