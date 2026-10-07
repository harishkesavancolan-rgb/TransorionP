"""/extract and /extract-batch also write each successful result to disk."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(autouse=True)
def _no_real_callback(monkeypatch):
    """Tests must never POST to the real callback API."""
    monkeypatch.setenv("EXTRACTOR_CALLBACK_URL", "")


def _result(name, valid):
    return {"source_file": name, "shipment_type": "export", "header": {}, "sheets": {"ITEM": []},
            "warnings": [], "validation": {"valid": valid}, "attempts": 1,
            "extraction_status": "validated" if valid else "validation_failed"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "SAVE_DIR", tmp_path / "chandru_api")

    async def fake_process(tmp_path_, filename, *a, **k):
        if filename.startswith("boom"):
            raise api.HTTPException(status_code=422, detail="no text")
        return _result(filename, valid=not filename.startswith("bad"))

    monkeypatch.setattr(api, "_process_pdf", fake_process)
    return TestClient(api.app)  # no `with`: skips the template-loading lifespan


def _pdf(name):
    return (name, b"%PDF-1.4 x", "application/pdf")


def test_extract_saves_a_valid_result_under_valid(client, tmp_path):
    r = client.post("/extract", files={"file": _pdf("inv1.pdf")})
    assert r.status_code == 200
    saved = json.loads((tmp_path / "chandru_api" / "valid" / "inv1.json").read_text(encoding="utf-8"))
    assert saved["source_file"] == "inv1.pdf" and saved == r.json()


def test_extract_saves_a_failed_result_under_validation_failed(client, tmp_path):
    client.post("/extract", files={"file": _pdf("bad1.pdf")})
    assert (tmp_path / "chandru_api" / "validation_failed" / "bad1.json").exists()
    assert not (tmp_path / "chandru_api" / "valid").exists()


def test_extract_batch_saves_every_successful_file_and_skips_errors(client, tmp_path):
    r = client.post("/extract-batch", files=[("files", _pdf("a.pdf")), ("files", _pdf("bad2.pdf")),
                                             ("files", _pdf("boom.pdf")), ("files", ("n.txt", b"x", "text/plain"))])
    assert r.status_code == 200 and len(r.json()["results"]) == 4
    root = tmp_path / "chandru_api"
    assert sorted(p.name for p in (root / "valid").iterdir()) == ["a.json"]
    assert sorted(p.name for p in (root / "validation_failed").iterdir()) == ["bad2.json"]


def test_same_filename_twice_in_one_batch_does_not_overwrite(client, tmp_path):
    client.post("/extract-batch", files=[("files", _pdf("dup.pdf")), ("files", _pdf("dup.pdf"))])
    assert sorted(p.name for p in (tmp_path / "chandru_api" / "valid").iterdir()) == ["dup.json", "dup_2.json"]


def test_a_path_in_the_upload_name_cannot_escape_the_folder(client, tmp_path):
    client.post("/extract", files={"file": _pdf("../../evil.pdf")})
    assert (tmp_path / "chandru_api" / "valid" / "evil.json").exists()
    assert not (tmp_path / "evil.json").exists()


def test_a_disk_error_does_not_break_the_response(client, tmp_path, monkeypatch):
    (tmp_path / "chandru_api").write_text("a FILE where the folder should be")
    r = client.post("/extract", files={"file": _pdf("inv2.pdf")})
    assert r.status_code == 200 and r.json()["source_file"] == "inv2.pdf"


# ── /save-pdf ────────────────────────────────────────────────────────────

@pytest.fixture
def pdf_client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RECEIVED_PDF_DIR", tmp_path / "received_pdfs")
    seen = {}

    async def fake_process(path, filename, model, requested_type, **k):
        seen[filename] = (path, requested_type)
        assert open(path, "rb").read().startswith(b"%PDF")  # extracting from the SAVED copy
        if filename.startswith("boom"):
            raise api.HTTPException(status_code=422, detail="no text")
        method = "ocrmypdf_docker" if filename.startswith("scan") else "2d_layout_canvas"
        return {**_result(filename, True), "extraction_method": method}

    monkeypatch.setattr(api, "_process_pdf", fake_process)
    c = TestClient(api.app)
    c.seen = seen
    return c


def test_save_pdf_saves_each_file_under_the_job_folder_and_reports_type_and_status(pdf_client, tmp_path):
    r = pdf_client.post("/save-pdf?job_id=123&shipment_type=import",
                        files=[("files", _pdf("a.pdf")), ("files", _pdf("scan1.pdf"))])
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == 123 and body["shipment_type"] == "import"
    a, s = body["results"]
    assert (a["job_id"], a["file_name"], a["document_type"], a["extraction_status"]) == (123, "a.pdf", "selectable", "success")
    assert (s["document_type"], s["extraction_status"]) == ("scanned", "success")
    assert a["extracted_json"]["source_file"] == "a.pdf"
    saved = sorted(p.name for p in (tmp_path / "received_pdfs" / "123").iterdir())
    assert saved == ["a.pdf", "scan1.pdf"]
    assert pdf_client.seen["a.pdf"][1] == "import"


def test_save_pdf_marks_failures_but_still_saves_the_pdf(pdf_client, tmp_path):
    r = pdf_client.post("/save-pdf?job_id=7&shipment_type=export",
                        files=[("files", _pdf("boom.pdf")), ("files", ("n.txt", b"x", "text/plain"))])
    boom, txt = r.json()["results"]
    assert boom["extraction_status"] == "failed" and boom["extracted_json"] is None and "no text" in boom["error"]
    assert boom["saved_path"] and boom["document_type"] is None
    assert txt["extraction_status"] == "failed" and txt["saved_path"] is None
    assert len(list((tmp_path / "received_pdfs" / "7").iterdir())) == 1  # only the PDF was stored


def test_save_pdf_keeps_the_original_name_and_a_same_name_twin_gets_a_suffix(pdf_client, tmp_path):
    r = pdf_client.post("/save-pdf?job_id=9&shipment_type=export", files=[("files", _pdf("d.pdf")), ("files", _pdf("d.pdf"))])
    assert sorted(p.name for p in (tmp_path / "received_pdfs" / "9").iterdir()) == ["d.pdf", "d_2.pdf"]
    first, second = r.json()["results"]
    assert first["saved_path"].endswith("d.pdf") and second["saved_path"].endswith("d_2.pdf")
    assert first["file_name"] == second["file_name"] == "d.pdf"


def test_save_pdf_requires_a_valid_shipment_type_and_job_id(pdf_client):
    assert pdf_client.post("/save-pdf?job_id=1&shipment_type=bogus", files=[("files", _pdf("a.pdf"))]).status_code == 422
    assert pdf_client.post("/save-pdf?shipment_type=import", files=[("files", _pdf("a.pdf"))]).status_code == 422


def test_save_pdf_saves_every_file_before_extracting_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RECEIVED_PDF_DIR", tmp_path / "received_pdfs")
    on_disk_at_extraction = []

    async def fake_process(path, filename, *a, **k):
        on_disk_at_extraction.append(sorted(p.name for p in (tmp_path / "received_pdfs" / "5").iterdir()))
        return {**_result(filename, True), "extraction_method": "2d_layout_canvas"}

    monkeypatch.setattr(api, "_process_pdf", fake_process)
    TestClient(api.app).post("/save-pdf?job_id=5&shipment_type=export",
                             files=[("files", _pdf("a.pdf")), ("files", _pdf("b.pdf")), ("files", _pdf("c.pdf"))])
    # at the moment the FIRST extraction starts, all three files are already saved
    assert on_disk_at_extraction[0] == ["a.pdf", "b.pdf", "c.pdf"]
    assert len(on_disk_at_extraction) == 3



# ── results callback ─────────────────────────────────────────────────────

def _mock_callback_client(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_save_pdf_posts_job_id_and_results_to_the_callback_api(pdf_client, monkeypatch):
    got = {}

    def handler(request):
        got["url"], got["headers"], got["body"] = str(request.url), request.headers, json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    _mock_callback_client(monkeypatch, handler)
    monkeypatch.setenv("EXTRACTOR_CALLBACK_URL", "http://cb.test/process_invoices/")
    monkeypatch.setenv("EXTRACTOR_CALLBACK_API_KEY", "k123")
    monkeypatch.setenv("EXTRACTOR_CALLBACK_BEARER_TOKEN", "tok")
    r = pdf_client.post("/save-pdf?job_id=5&shipment_type=export",
                        files=[("files", _pdf("a.pdf")), ("files", _pdf("boom.pdf"))])
    assert got["url"] == "http://cb.test/process_invoices/"
    assert got["headers"]["x-api-key"] == "k123" and got["headers"]["authorization"] == "Bearer tok"
    assert got["body"]["job_id"] == 5
    ok, failed = got["body"]["results"]
    assert ok["source_file"] == "a.pdf" and "sheets" in ok and "header" in ok   # the extraction result itself
    assert ok["document_type"] == "selectable"
    assert failed == {"source_file": "boom.pdf", "document_type": None, "error": "no text"}
    assert r.json()["callback"] == {"sent": True, "status_code": 200}


def test_callback_carries_selectable_or_scanned_per_file(pdf_client, monkeypatch):
    got = {}

    def handler(request):
        got["body"] = json.loads(request.content)
        return httpx.Response(200)

    _mock_callback_client(monkeypatch, handler)
    monkeypatch.setenv("EXTRACTOR_CALLBACK_URL", "http://cb.test/x/")
    pdf_client.post("/save-pdf?job_id=5&shipment_type=export", files=[("files", _pdf("a.pdf")), ("files", _pdf("scan1.pdf"))])
    assert [r["document_type"] for r in got["body"]["results"]] == ["selectable", "scanned"]


def test_a_failing_callback_does_not_break_the_response(pdf_client, monkeypatch):
    _mock_callback_client(monkeypatch, lambda request: httpx.Response(401, text="bad token"))
    monkeypatch.setenv("EXTRACTOR_CALLBACK_URL", "http://cb.test/x/")
    r = pdf_client.post("/save-pdf?job_id=5&shipment_type=export", files=[("files", _pdf("a.pdf"))])
    assert r.status_code == 200 and r.json()["results"][0]["extraction_status"] == "success"
    assert r.json()["callback"]["sent"] is False and r.json()["callback"]["status_code"] == 401


def test_an_unreachable_callback_is_reported_not_raised(pdf_client, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("no route to host")

    _mock_callback_client(monkeypatch, handler)
    monkeypatch.setenv("EXTRACTOR_CALLBACK_URL", "http://cb.test/x/")
    r = pdf_client.post("/save-pdf?job_id=5&shipment_type=export", files=[("files", _pdf("a.pdf"))])
    assert r.status_code == 200 and "no route" in r.json()["callback"]["error"]


def test_callback_is_off_when_the_url_is_empty(pdf_client):
    r = pdf_client.post("/save-pdf?job_id=5&shipment_type=export", files=[("files", _pdf("a.pdf"))])
    assert r.json()["callback"]["sent"] is False and "disabled" in r.json()["callback"]["reason"]
