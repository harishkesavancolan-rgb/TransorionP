"""
Tests for llm_extract.apply_header_guardrails() (the deterministic
post-processing guardrails run after both LLM passes) and
extract_line_items_chunked() (the page-chunked line-items pass).
apply_header_guardrails() is synchronous and mutates its header dict in
place, so it's exercised directly, no mocking needed; the chunked
extraction's own LLM call (extract_line_items) is mocked so those tests
run instantly and free -- they check the chunking/merge logic, not model
output quality.

Uses asyncio.run() directly rather than pytest-asyncio, same as
test_reextraction.py.
"""
from __future__ import annotations

import asyncio

import llm_extract

_USAGE = {"input_tokens": 10, "cached_tokens": 0, "output_tokens": 10}


# ── total_packages: narrative packing-list fallback ─────────────────────

def test_total_packages_recovered_from_narrative_carton_description():
    # Model copied the piece quantity (3840) into total_packages instead
    # of the real package count (192 carton boxes); 4 is a pallet count
    # that must NOT be picked up either.
    text = "Four Nos Of Pallet Containing 3840 Nos (20 Nos Each in 192 carton boxes)\n"
    header = {"total_packages": 3840}
    items = [{"item_ser_no": 1, "quantity": 3840}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["total_packages"] == 192


def test_total_packages_recovered_when_missing_entirely():
    text = "Packed in 45 cartons, net weight 900 KG\n"
    header = {"total_packages": None}
    items = [{"item_ser_no": 1, "quantity": 900}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["total_packages"] == 45


def test_total_packages_left_alone_when_already_different_and_plausible():
    # Not the quantity and not missing -- an already-distinct value is
    # trusted rather than second-guessed against text.
    text = "Packed in 45 cartons\n"
    header = {"total_packages": 12}
    items = [{"item_ser_no": 1, "quantity": 900}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["total_packages"] == 12


def test_total_packages_not_guessed_from_ambiguous_multiple_candidates():
    # Two different, conflicting package-unit mentions -- no way to pick
    # the right one from shape alone, so it's left as-is rather than
    # guessed.
    text = "192 cartons on pallet A, 50 cases on pallet B\n"
    header = {"total_packages": 3840}
    items = [{"item_ser_no": 1, "quantity": 3840}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["total_packages"] == 3840


def test_total_packages_ignores_pallet_count():
    # No carton/case/box/package wording at all, only a pallet count --
    # must NOT be substituted in as total_packages.
    text = "Shipped as 4 pallets\n"
    header = {"total_packages": 3840}
    items = [{"item_ser_no": 1, "quantity": 3840}]
    llm_extract.apply_header_guardrails(header, items, text)
    assert header["total_packages"] == 3840


def test_apply_header_guardrails_is_a_noop_on_a_non_dict_header():
    # Defensive: a malformed LLM response (header came back as something
    # other than a dict) must not raise.
    llm_extract.apply_header_guardrails(None, [], "some text\n")
    llm_extract.apply_header_guardrails("not a dict", [], "some text\n")


# ── _split_text_by_page ──────────────────────────────────────────────────

def _page_marker(idx, total, name="inv.pdf"):
    return f"\n{'=' * 35} PAGE {idx} OF {total} ({name}) {'=' * 35}\n"


def test_split_text_by_page_splits_on_markers():
    text = (
        _page_marker(1, 3) + "page one content\n"
        + _page_marker(2, 3) + "page two content\n"
        + _page_marker(3, 3) + "page three content\n"
    )
    pages = llm_extract._split_text_by_page(text)
    assert len(pages) == 3
    assert "PAGE 1 OF 3" in pages[0] and "page one content" in pages[0]
    assert "PAGE 2 OF 3" in pages[1] and "page two content" in pages[1]
    assert "PAGE 3 OF 3" in pages[2] and "page three content" in pages[2]


def test_split_text_by_page_single_page_returns_unchanged():
    text = _page_marker(1, 1) + "only page\n"
    pages = llm_extract._split_text_by_page(text)
    assert pages == [text]


def test_split_text_by_page_no_markers_returns_unchanged():
    text = "plain text with no page markers at all\n"
    pages = llm_extract._split_text_by_page(text)
    assert pages == [text]


# ── extract_line_items_chunked ───────────────────────────────────────────

def test_chunked_extraction_below_threshold_makes_a_single_call(monkeypatch):
    calls = []

    async def fake_extract_line_items(text, **kwargs):
        calls.append(text)
        return [{"item_ser_no": 1}], dict(_USAGE)

    monkeypatch.setattr(llm_extract, "extract_line_items", fake_extract_line_items)

    text = (
        _page_marker(1, 2) + "page one\n"
        + _page_marker(2, 2) + "page two\n"
    )
    items, usage = asyncio.run(llm_extract.extract_line_items_chunked(text, model="gpt-5-nano"))
    assert len(calls) == 1  # 2 pages <= threshold -- single call, whole text
    assert calls[0] == text
    assert items == [{"item_ser_no": 1}]
    assert usage == _USAGE


def test_chunked_extraction_above_threshold_splits_and_merges_per_page(monkeypatch):
    calls = []

    async def fake_extract_line_items(text, **kwargs):
        calls.append(text)
        # Return one item per call, tagged by which page text it saw.
        page_num = text.split("PAGE ")[1].split(" ")[0]
        return [{"item_ser_no": int(page_num), "product_description": f"item from page {page_num}"}], dict(_USAGE)

    monkeypatch.setattr(llm_extract, "extract_line_items", fake_extract_line_items)

    n_pages = 5  # > _LINE_ITEMS_CHUNK_PAGE_THRESHOLD (3)
    text = "".join(_page_marker(i, n_pages) + f"content {i}\n" for i in range(1, n_pages + 1))

    items, usage = asyncio.run(llm_extract.extract_line_items_chunked(text, model="gpt-5-nano"))

    assert len(calls) == n_pages  # one call per page, not one call for the whole doc
    assert [it["item_ser_no"] for it in items] == [1, 2, 3, 4, 5]  # merged in page order
    # Usage summed across all per-page calls, not just the last one.
    assert usage["input_tokens"] == _USAGE["input_tokens"] * n_pages
    assert usage["output_tokens"] == _USAGE["output_tokens"] * n_pages


def test_chunked_extraction_passes_feedback_to_every_chunk(monkeypatch):
    seen_feedback = []

    async def fake_extract_line_items(text, feedback=None, **kwargs):
        seen_feedback.append(feedback)
        return [], dict(_USAGE)

    monkeypatch.setattr(llm_extract, "extract_line_items", fake_extract_line_items)

    n_pages = 4
    text = "".join(_page_marker(i, n_pages) + f"content {i}\n" for i in range(1, n_pages + 1))
    asyncio.run(llm_extract.extract_line_items_chunked(text, model="gpt-5-nano", feedback="fix the totals"))

    assert seen_feedback == ["fix the totals"] * n_pages
