"""
pdf_reader.py
-------------
Extracts text AND tables from an invoice PDF using spatial layout preservation.

Layout text extraction: Uses `pdfplumber` with `layout=True` to reconstruct
the 2D visual layout of the PDF page from character/word coordinates.
This preserves columns, tables, and spacing so the LLM has exact spatial
awareness.

Dual-stream view: Uses `pypdfium2` (Apache-2.0 / BSD-3-Clause) for sequential
text extraction to provide a column-separated stream view alongside the
pdfplumber layout view. This prevents garbled side-by-side columns
(e.g. interleaving "Tesla" + "Tesla" → "TTeessllaa").

Table extraction: Uses `pdfplumber.extract_tables()` to detect cell grids,
rendered as formatted markdown tables.

OCR fallback: When a PDF is purely a scanned image with no embedded text,
falls back to pypdfium2 rasterization + RapidOCR / PaddleOCR. Every tier of
this fallback chain logs why it was or wasn't used -- a silently-skipped
tier (Docker not installed, an OCR engine failing to import) used to be
invisible; now it shows up in the logs instead of just a quality drop.

License: All PDF libraries used here are permissively licensed:
  - pdfplumber: MIT
  - pypdfium2: Apache-2.0 / BSD-3-Clause (Google PDFium)
  - rapidocr_onnxruntime: Apache-2.0
"""
from __future__ import annotations
from pathlib import Path

import logging
import math
import os
import re
import shutil
import statistics

import pdfplumber
import pypdfium2 as pdfium

try:
    import scan_validator
except Exception:
    scan_validator = None

logger = logging.getLogger("invoice_extractor")

# Tilt beyond this is treated as "too far off to trust a simple in-plane
# rotation correction" for a selectable PDF -- warn instead of correcting.
_MAX_CORRECTABLE_SKEW_DEG = 30.0
# Below this, correction is a no-op not worth the (tiny) extra computation.
_MIN_SKEW_TO_CORRECT_DEG = 0.3


# Marks a detected column boundary in the 2D layout canvas -- a box-drawing
# character chosen specifically because it never appears in real invoice
# text, unlike "|" which occasionally shows up in part numbers/codes.
_COLUMN_BOUNDARY_MARKER = "│"  # "│"

# How close two same-character glyphs need to sit to count as the same
# duplicated glyph rather than two genuinely different adjacent characters.
_DUPLICATE_GLYPH_TOLERANCE_PT = 1.0


def _dedupe_overlapping_chars(
    chars: list[dict],
    x_tolerance: float = _DUPLICATE_GLYPH_TOLERANCE_PT,
    top_tolerance_factor: float = 0.35,
) -> list[dict]:
    """
    Some invoices' embedded text layer contains two near-identical copies of
    the same content, rendered at slightly different font scales but at
    almost the same position (confirmed on real files: e.g. two 'i'
    characters at the same (top, x0), one at 5.26pt tall, one at 6.89pt).
    `page.extract_words()` doesn't know these are duplicates -- it just
    clusters nearby characters by proximity, so it glues interleaved
    fragments of BOTH copies into garbage tokens (e.g. "provisions," comes
    out as "pp roo vv ii ss iio onn ss ii nn cc ll uu..."), which then
    scatters across line-grouping since the two copies' positions aren't
    quite identical. This has to be fixed before word-clustering even
    happens -- there's no way to un-glue "pp" back into a single "p" once
    extract_words() has already merged the duplicate into one token.

    Matched by POSITION ALONE, not by decoded text (confirmed on a real
    file: an overlaid duplicate line used a second embedded font whose cmap
    decoded the same visual glyph differently -- one copy's 'y' came out as
    text 'y', the other copy's came out as 'v', same for ',' vs '.' -- so a
    dedup keyed on `text == text` never recognized them as duplicates, both
    copies survived, and extract_words() shredded the interleaved result
    into single-letter tokens). Requiring text equality was never actually
    needed for safety either: two genuinely different, unrelated characters
    don't land on virtually the same point -- the tightest observed gap
    between distinct adjacent glyphs on real invoices here is ~2.6pt in x0,
    several times the tolerance used below.

    The vertical tolerance scales with glyph height rather than using a
    flat value: two overlaid copies at different font sizes don't sit at
    *exactly* the same baseline (confirmed on a real file: one copy's
    baseline was 1.65pt below the other's, comfortably more than a flat
    1.0pt tolerance would allow but well under a real line-to-line pitch on
    this size of text).

    Known limitation: the offset between overlaid copies isn't consistent
    even within one document -- a different duplicated line on the same
    page, same font, was 3.69pt apart, wider than this tolerance catches.
    Widening the factor to reach it was tried and reverted: at that width
    it started deleting characters from the CORRECT copy instead (matching
    one of its glyphs against the wrong nearby duplicate and dropping it),
    which is worse than leaving a readable correct line next to a garbled
    decoy. When the two overlaid copies also disagree on the actual
    content (confirmed on a real file: a duplicated "District of Origin"
    line read 12 in one copy and 42 in the other), position alone can't
    safely resolve which copy is real -- that needs a value-level rule
    downstream, not a wider position tolerance here.

    Keeps the first-encountered copy of each near-duplicate (chars are
    processed in top-to-bottom, left-to-right order, so true duplicates --
    which sit almost on top of each other -- end up adjacent in that order;
    a small lookback window is enough to catch them without an O(n^2) scan).

    A whitespace character is never matched against a non-whitespace one,
    even if their positions are within tolerance -- confirmed on a real
    invoice using a condensed font ("Aptos Narrow, Bold"): the space
    between "SHIP" and "FROM:" sat only 0.96pt before the "F", tighter
    than this function's own 1.0pt x-tolerance (justified elsewhere by a
    ~2.6pt minimum gap between distinct glyphs -- true for the invoices
    that assumption came from, not for this condensed one). Since the
    space sorts first at that position, it got kept and the real "F" that
    followed was flagged as its "duplicate" and dropped -- silently
    eating the first letter of every single word on the page ("SHIP
    FROM:" -> "SHIP ROM:", "HYVE Solutions" -> "HYVE olutions", ...). The
    duplicate-text-layer bug this function targets always duplicates a
    VISIBLE glyph, never a space, so excluding whitespace-vs-non-whitespace
    pairs closes this false-positive without narrowing the tolerance that
    the genuine duplicate-glyph cases rely on.
    """
    chars_sorted = sorted(chars, key=lambda c: (c["top"], c["x0"]))
    kept: list[dict] = []
    for c in chars_sorted:
        is_dup = False
        c_is_space = c["text"].isspace()
        for k in kept[-12:]:
            if c_is_space != k["text"].isspace():
                continue
            height = min(k["bottom"] - k["top"], c["bottom"] - c["top"]) or 1.0
            top_tolerance = max(x_tolerance, height * top_tolerance_factor)
            if (
                abs(k["top"] - c["top"]) <= top_tolerance
                and abs(k["x0"] - c["x0"]) <= x_tolerance
            ):
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
    return kept


def _detect_column_boundaries(
    lines: list[list[dict]],
    page: pdfplumber.page.Page,
    char_width_pts: float,
    min_gap_multiplier: float = 4.0,
    min_recurrence_fraction: float = 0.15,
    min_recurrence_count: int = 3,
) -> list[float]:
    """
    Returns x-positions (in points) that are confirmed table-column
    boundaries on this page, from two signals:

    1. Recurring wide gaps: a horizontal gap between two adjacent words on
       a line only counts as a real column boundary if a gap shows up at
       roughly the *same x-position across multiple lines* -- that's the
       actual signature of a table column, as opposed to one line just
       happening to have a wide gap (e.g. trailing spaces after a label
       like "Total:      "). A single wide gap on one line is ignored.

       Each qualifying gap contributes two candidate x-positions, not just
       its midpoint: the midpoint itself, and the next word's left edge
       (pulled in by one char width so it sits inside the gap rather than
       on the text). The midpoint alone misses a common two-column layout
       -- a label/address block whose LEFT column is ragged, varying in
       width row to row (confirmed on a real file: "Tesla, Inc." vs.
       "NA-US-TX-Kyle-201 Logistics Dr (Kyle 2- GA1)" vs. "201 Logistics
       Drive" as consecutive rows of the same left column) -- so the gap's
       midpoint shifts on nearly every row and never recurs at one
       x-position even though a real second column sits to the right of
       it. What stays put row to row is the RIGHT column's left edge, so
       that's tracked as its own candidate and confirmed the same way.
    2. Ruled vertical lines the document itself drew (page.lines /
       page.rects) -- when there's a real column divider printed on the
       page, trust it directly; it doesn't need to recur across lines.
    """
    cluster_tolerance_pts = char_width_pts * 2
    min_gap_pts = char_width_pts * min_gap_multiplier

    # 1. Collect wide-gap midpoints AND next-word left-edges from inter-word
    # spacing on each line (see docstring for why both are needed).
    gap_midpoints: list[float] = []
    for line in lines:
        for prev_w, next_w in zip(line, line[1:]):
            gap = next_w["x0"] - prev_w["x1"]
            if gap >= min_gap_pts:
                gap_midpoints.append((prev_w["x1"] + next_w["x0"]) / 2.0)
                gap_midpoints.append(next_w["x0"] - char_width_pts)

    # Clustered against each band's FIRST (leftmost) member, not its most
    # recently added one -- comparing to the last element lets a band drift
    # arbitrarily far via a chain of small steps (each under tolerance from
    # its immediate neighbor, but the band's overall span ends up far wider
    # than cluster_tolerance_pts), which silently merges two genuinely
    # different columns into one averaged, inaccurate boundary (confirmed
    # on a real file: an address block's ~241pt column and a metadata
    # table's ~248pt column chained together this way into a single ~246pt
    # average that fell outside BOTH real gaps).
    gap_midpoints.sort()
    bands: list[list[float]] = []
    for x in gap_midpoints:
        if bands and x - bands[-1][0] <= cluster_tolerance_pts:
            bands[-1].append(x)
        else:
            bands.append([x])

    min_recurrence = max(min_recurrence_count, int(len(lines) * min_recurrence_fraction))
    confirmed = [sum(band) / len(band) for band in bands if len(band) >= min_recurrence]

    # 2. Ruled vertical lines -- a tall, thin line/rect is a drawn column
    # divider. Always trusted, no recurrence needed.
    min_rule_height = page.height * 0.05
    for line_obj in page.lines or []:
        if abs(line_obj["x0"] - line_obj["x1"]) <= 1.0 and (line_obj["bottom"] - line_obj["top"]) >= min_rule_height:
            confirmed.append(line_obj["x0"])
    for rect in page.rects or []:
        if rect["width"] <= 1.5 and rect["height"] >= min_rule_height:
            confirmed.append(rect["x0"])

    return sorted(set(round(x, 1) for x in confirmed))


def _gap_crosses_boundary(gap_start_pts: float, gap_end_pts: float, boundaries: list[float]) -> bool:
    return any(gap_start_pts <= b <= gap_end_pts for b in boundaries)


def _deskew_words(words: list[dict], skew_deg: float, page_width: float, page_height: float) -> list[dict]:
    """
    Rotates each word's CENTER point by -skew_deg around the page center to
    undo a measured tilt, keeping the word's own width/height unchanged --
    so the line-grouping/rendering below (which assumes horizontal lines of
    text) sees an effectively upright page.

    Deliberately rotates the center point only, not the four corners of
    each word's bounding box: line-grouping only ever checks a word's
    vertical CENTER against a tolerance, and rendering only cares about its
    x0 for column position, so the center is the only thing that needs to
    be geometrically correct. Rotating the four corners and re-deriving an
    axis-aligned bbox from them (the first version of this function) is
    NOT an exact inverse of anything and measurably overshoots for wide
    words: a typical word here is ~19pt wide but only ~8pt tall, and a
    wide-but-short box's corners swing much further vertically than
    horizontally under rotation, so bounding-box-of-rotated-corners
    inflates a word's vertical extent well beyond what a correction should
    introduce (confirmed empirically: 11-21pt of spurious vertical error
    at just 8-15 degrees of tilt, more than enough to break line-grouping's
    own ~2-4pt tolerance -- i.e. the "fix" would have re-broken the exact
    thing it was correcting). Center-point rotation has no such artifact:
    rotating a point by -theta then by +theta is an exact identity.

    This corrects the coordinate SPACE, not any image -- there's no
    re-rendering or re-OCR involved, which is the whole point: it's cheap
    enough to run on every page, unlike rasterizing+re-OCRing a selectable
    PDF just to fix a few degrees of tilt.

    Returns new word dicts (originals are untouched).
    """
    theta = math.radians(-skew_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = page_width / 2.0, page_height / 2.0

    corrected: list[dict] = []
    for w in words:
        width, height = w["x1"] - w["x0"], w["bottom"] - w["top"]
        center_x, center_y = (w["x0"] + w["x1"]) / 2.0, (w["top"] + w["bottom"]) / 2.0
        dx, dy = center_x - cx, center_y - cy
        new_cx = cx + dx * cos_t - dy * sin_t
        new_cy = cy + dx * sin_t + dy * cos_t

        nw = dict(w)
        nw["x0"], nw["x1"] = new_cx - width / 2.0, new_cx + width / 2.0
        nw["top"], nw["bottom"] = new_cy - height / 2.0, new_cy + height / 2.0
        corrected.append(nw)
    return corrected


def extract_page_layout_canvas(
    page: pdfplumber.page.Page,
    char_width_pts: float | None = None,
    v_tolerance_factor: float = 0.35,
    skew_deg: float | None = None,
) -> str:
    """
    Extracts complete text from a PDF page onto a 2D spatial canvas.
    Preserves exact linearity, columns, and spacing without word collisions.

    Confirmed table-column boundaries (see _detect_column_boundaries) get
    an explicit marker (_COLUMN_BOUNDARY_MARKER) instead of plain
    whitespace, wherever a line's own gap actually crosses one -- so a
    model reading the flattened text can tell "two separate columns with
    a wide gap between them" apart from "one long run-on sentence with
    incidental extra spacing," which plain whitespace can't distinguish.

    `skew_deg`, when given and within a correctable range, is used to
    de-skew word coordinates (see _deskew_words) before line-grouping --
    otherwise a tilted page's line-grouping (which assumes horizontal
    lines) silently misgroups words the farther apart they are
    horizontally, since the vertical drift from tilt scales with distance.

    Characters are deduplicated (see _dedupe_overlapping_chars) BEFORE
    word extraction -- some invoices' embedded text layer has two
    near-identical overlapping copies of the same content, and
    extract_words() would otherwise glue interleaved fragments of both
    copies into garbage tokens that can't be un-glued afterward.
    """
    chars = _dedupe_overlapping_chars(page.chars)
    words = pdfplumber.utils.extract_words(chars, x_tolerance=1.5, y_tolerance=1.5)
    if not words:
        return ""

    if (
        skew_deg is not None
        and _MIN_SKEW_TO_CORRECT_DEG <= abs(skew_deg) <= _MAX_CORRECTABLE_SKEW_DEG
    ):
        words = _deskew_words(words, skew_deg, page.width, page.height)

    # 1. Adaptive character width from page font metrics
    if char_width_pts is None:
        if page.chars:
            valid_widths = [c['width'] for c in page.chars if c.get('text', '').strip()]
            char_width_pts = round(statistics.median(valid_widths), 2) if valid_widths else 4.0
        else:
            char_width_pts = 4.0

    # 2. Adaptive vertical tolerance based on font heights
    font_heights = [w['bottom'] - w['top'] for w in words]
    med_height = statistics.median(font_heights) if font_heights else 8.0
    v_tolerance = max(2.0, med_height * v_tolerance_factor)

    # 3. Sort words top-to-bottom, left-to-right
    sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))

    # 4. Group into horizontal lines
    lines: list[list[dict]] = []
    curr_line: list[dict] = []
    curr_center: float | None = None

    for w in sorted_words:
        w_center = (w['top'] + w['bottom']) / 2.0
        if curr_center is None:
            curr_center = w_center
            curr_line.append(w)
        elif abs(w_center - curr_center) <= v_tolerance:
            curr_line.append(w)
            curr_center = sum((x['top'] + x['bottom']) / 2.0 for x in curr_line) / len(curr_line)
        else:
            lines.append(sorted(curr_line, key=lambda x: x['x0']))
            curr_line = [w]
            curr_center = w_center

    if curr_line:
        lines.append(sorted(curr_line, key=lambda x: x['x0']))

    # 4b. Detect recurring/ruled column boundaries across the whole page
    # before rendering -- rendering needs to know about them per-line.
    column_boundaries = _detect_column_boundaries(lines, page, char_width_pts)

    # 5. Render onto 2D character canvas with collision prevention
    total_cols = int(math.ceil(page.width / char_width_pts)) + 20
    output_rows: list[str] = []

    for line in lines:
        row = [' '] * total_cols
        curr_col = 0
        for idx, w in enumerate(line):
            target_col = int(round(w['x0'] / char_width_pts))
            # Collision prevention: advance if target is already occupied
            col = max(target_col, curr_col + 1 if curr_col > 0 else target_col)

            # If THIS line's own gap from the previous word is itself wide
            # (not just a normal single space) AND that gap crosses a
            # confirmed column boundary, mark it explicitly instead of
            # leaving it as plain whitespace. Both conditions matter: a
            # confirmed boundary is a single clustered x-position, and
            # checking overlap alone would flag any narrow, ordinary
            # inter-word gap that happens to straddle that x by
            # coincidence (e.g. "Doddanakundi Industrial" -- a normal
            # ~2pt space -- got wrongly split this way before this check
            # was added, just because a table's column gap elsewhere on
            # the page averaged to an x-position that fell inside it).
            if idx > 0 and column_boundaries:
                prev_w = line[idx - 1]
                this_gap = w["x0"] - prev_w["x1"]
                if this_gap >= char_width_pts * 1.5 and _gap_crosses_boundary(prev_w["x1"], w["x0"], column_boundaries):
                    delim_col = (curr_col + col) // 2
                    if curr_col <= delim_col < len(row):
                        row[delim_col] = _COLUMN_BOUNDARY_MARKER

            w_text = w['text']
            for i, ch in enumerate(w_text):
                pos = col + i
                if pos < len(row):
                    row[pos] = ch
            curr_col = col + len(w_text)

        output_rows.append(''.join(row).rstrip())

    return '\n'.join(output_rows)


def _detect_page_skew(pdf_path: str | Path, page_index: int) -> float | None:
    """Best-effort tilt estimate for one page via scan_validator (Hough-line
    based, works on rasterized pixels regardless of scanned vs. selectable).
    Returns None if scan_validator isn't available or the check fails --
    callers should treat that as "no correction," not an error."""
    if scan_validator is None:
        return None
    try:
        return scan_validator.validate_pdf(str(pdf_path), page=page_index)["skew_deg"]
    except Exception as e:
        logger.debug("%s page %d: skew detection failed (%s)", pdf_path, page_index, e)
        return None


def extract_full_pdf(pdf_path: str | Path, char_width_pts: float | None = None) -> str:
    """
    Extracts the ENTIRE PDF document (all pages) preserving complete wordings,
    linearity, column separation, and spacing.
    Only includes pages that contain actual text characters (scanned pages with 0 chars are skipped).

    Each page's tilt is estimated (see _detect_page_skew) and corrected in
    coordinate space before line-grouping (see extract_page_layout_canvas),
    so a moderately tilted selectable PDF doesn't need a full re-render+OCR
    pass just to read its lines correctly.
    """
    pdf_path = Path(pdf_path)
    extracted_pages: list[str] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        num_pages = len(pdf.pages)
        for idx, page in enumerate(pdf.pages, start=1):
            skew_deg = _detect_page_skew(pdf_path, idx - 1)
            page_text = extract_page_layout_canvas(page, char_width_pts=char_width_pts, skew_deg=skew_deg)
            if page_text and page_text.strip():
                header = f"\n{'=' * 35} PAGE {idx} OF {num_pages} ({pdf_path.name}) {'=' * 35}\n"
                extracted_pages.append(header + page_text)

    return "\n".join(extracted_pages).strip()


# Regex for genuine words containing at least one alphanumeric character --
# shared by every per-page text-density check below.
_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]")


def _page_text_stats(pdf_path: str | Path, min_word_len: int = 2) -> list[dict[str, int]]:
    """
    Opens the PDF ONCE and returns per-page text-density stats:
        [{"chars": int, "valid_words": int, "single_char_words": int}, ...]

    Both is_scanned_pdf() and the --force-ocr/--skip-text decision need
    per-page word/char counts -- this is the one place that computes them,
    so neither re-parses the document or keeps a second definition of
    "what counts as a real word on a page."

    Chars are deduplicated first (see _dedupe_overlapping_chars) -- a page
    with a duplicated/overlapping text layer would otherwise inflate both
    the char count and single_char_words (interleaved duplicate glyphs
    routinely extract as garbled single-character tokens), skewing the
    scanned-vs-digital and force-ocr decisions that read these stats.
    """
    stats: list[dict[str, int]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            chars = _dedupe_overlapping_chars(page.chars)
            words = pdfplumber.utils.extract_words(chars)
            page_text = pdfplumber.utils.extract_text(chars) or ""

            valid_words = 0
            single_char_words = 0
            for w in words:
                t = w.get("text", "").strip()
                if _WORD_PATTERN.search(t):
                    if len(t) >= min_word_len:
                        valid_words += 1
                    else:
                        # Catch single-character vertical letter noise (like D \n R \n E)
                        single_char_words += 1

            stats.append({
                "chars": len(page_text.strip()),
                "valid_words": valid_words,
                "single_char_words": single_char_words,
            })
    return stats


def is_scanned_pdf(
    pdf_path: str | Path,
    min_words_per_page: int = 25,
    min_word_len: int = 2,
    min_avg_chars_per_page: int = 150,
) -> bool:
    """
    Strictly evaluates whether a PDF is a scanned image (or has garbage OCR
    layers) versus a true readable digital PDF, using PER-PAGE density
    checks rather than a single whole-document character count.

    That distinction matters: extract_text()'s old gate was just "does the
    whole PDF have >= 40 characters of embedded text anywhere" -- a
    multi-page scanned invoice with nothing but a letterhead or a single
    stamped cover line clears that easily while every other page is a pure
    image, so the whole document got (wrongly) treated as fully digital.
    Checking density page-by-page catches that.

    Returns:
        True  -> PDF is SCANNED (or corrupted/sparse text layer) -> route to OCR.
        False -> PDF is a true digital PDF with dense, readable text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return True

    try:
        page_stats = _page_text_stats(pdf_path, min_word_len=min_word_len)
    except Exception as e:
        logger.debug(
            "%s: is_scanned_pdf check failed (%s), treating as scanned to trigger OCR",
            pdf_path, e,
        )
        return True

    total_pages = len(page_stats)
    if total_pages == 0:
        return True

    total_chars = sum(p["chars"] for p in page_stats)
    total_valid_words = sum(p["valid_words"] for p in page_stats)
    single_char_word_count = sum(p["single_char_words"] for p in page_stats)
    pages_with_dense_text = sum(1 for p in page_stats if p["valid_words"] >= min_words_per_page)

    # 1. Average characters per page (commercial invoices run ~300-2000 chars/page)
    avg_chars_per_page = total_chars / total_pages
    if avg_chars_per_page < min_avg_chars_per_page:
        logger.debug(
            "%s: avg %.0f chars/page < %d -> scanned",
            pdf_path, avg_chars_per_page, min_avg_chars_per_page,
        )
        return True

    # 2. Average real words per page
    avg_words_per_page = total_valid_words / total_pages
    if avg_words_per_page < min_words_per_page:
        logger.debug(
            "%s: avg %.1f words/page < %d -> scanned",
            pdf_path, avg_words_per_page, min_words_per_page,
        )
        return True

    # 3. Vertical fragmentation / garbage letter noise (rotated or corrupted text layer)
    if single_char_word_count > (total_valid_words * 0.8):
        logger.debug(
            "%s: %d single-char word fragments vs %d real words -> scanned/corrupted",
            pdf_path, single_char_word_count, total_valid_words,
        )
        return True

    # 4. In multi-page PDFs, at least half the pages should have meaningful content
    if total_pages > 1 and pages_with_dense_text < (total_pages / 2):
        logger.debug(
            "%s: only %d/%d pages have dense text -> scanned",
            pdf_path, pages_with_dense_text, total_pages,
        )
        return True

    return False


def _has_partial_text_pages(
    pdf_path: str | Path,
    min_words_per_page: int = 25,
    min_word_len: int = 2,
) -> bool:
    """
    True if any page has a text layer that isn't dense/real enough --
    the specific case OCRmyPDF's --skip-text mishandles.

    The check is `chars > 0 and valid_words < min_words_per_page`, NOT
    `0 < valid_words < min_words_per_page` -- those are not the same thing,
    and the difference matters. A real page found in production: 4674 raw
    characters (`page.extract_text()`), but 0 "valid words" by our
    alphanumeric-word-of-length>=2 definition -- i.e. pdfplumber's word
    extractor found nothing it considered a real word, but there plainly
    *was* a text layer there (garbled/corrupted encoding, not a blank
    page). `valid_words == 0` alone made this look like a pure image page
    ("nothing there for --skip-text to skip"), so --skip-text was
    auto-selected -- and it wrongly skipped the page anyway, because
    OCRmyPDF's own "does this page already have text" check isn't
    word-quality-aware either; it just sees *a* text layer and bails.
    Forcing OCR on that exact file recovered the real content.

    So: a page only counts as "nothing here, --skip-text is safe" when it
    has ZERO extracted characters at all, not merely zero recognizable
    words. Any non-zero character count below the dense-page word bar is
    "partial" and should force OCR.

    Still reuses the exact same per-page stats and the same
    min_words_per_page bar as is_scanned_pdf() -- just testing `chars`
    instead of `valid_words` for the "is there truly nothing here" leg.
    """
    try:
        page_stats = _page_text_stats(pdf_path, min_word_len=min_word_len)
    except Exception as e:
        logger.debug(
            "%s: partial-text-page check failed (%s), defaulting to --force-ocr to be safe",
            pdf_path, e,
        )
        return True

    return any(p["chars"] > 0 and p["valid_words"] < min_words_per_page for p in page_stats)


# Global OCR engine cache so neural network models load once into memory,
# saving 5-10 seconds on every subsequent PDF extraction.
_OCR_ENGINE = None
_OCR_ENGINE_TYPE = None


def _get_ocr_engine():
    """Initializes and caches the fastest available pure-Python OCR engine."""
    global _OCR_ENGINE, _OCR_ENGINE_TYPE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE, _OCR_ENGINE_TYPE

    # Disable buggy experimental Paddle PIR/OneDNN compiler on Windows CPU
    try:
        import paddle
        paddle.set_flags({"FLAGS_enable_pir_api": 0})
    except Exception as e:
        logger.debug("Paddle PIR flag tweak skipped (paddle not installed or old version): %s", e)

    # 1. Try RapidOCR first (rock-solid ONNX runtime on Windows CPU, ~1s per page)
    try:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
        _OCR_ENGINE_TYPE = "rapidocr"
        logger.info("OCR engine: using RapidOCR")
        return _OCR_ENGINE, _OCR_ENGINE_TYPE
    except Exception as e:
        logger.debug("RapidOCR unavailable, trying PaddleOCR next: %s", e)

    # 2. Try standard PaddleOCR with OneDNN/MKLDNN disabled to prevent Windows C++ crashes
    try:
        from paddleocr import PaddleOCR
        try:
            _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)
        except Exception:
            try:
                _OCR_ENGINE = PaddleOCR(lang="en", enable_mkldnn=False)
            except Exception:
                _OCR_ENGINE = PaddleOCR(lang="en")
        _OCR_ENGINE_TYPE = "paddleocr"
        logger.info("OCR engine: using PaddleOCR")
        return _OCR_ENGINE, _OCR_ENGINE_TYPE
    except Exception as e:
        logger.debug("PaddleOCR unavailable, trying PaddleOCRVL next: %s", e)

    # 3. Try PaddleOCRVL (heavy VL pipeline)
    try:
        from paddleocr import PaddleOCRVL
        try:
            _OCR_ENGINE = PaddleOCRVL(pipeline_version="v1.5")
        except Exception:
            _OCR_ENGINE = PaddleOCRVL()
        _OCR_ENGINE_TYPE = "paddleocr_vl"
        logger.info("OCR engine: using PaddleOCRVL")
        return _OCR_ENGINE, _OCR_ENGINE_TYPE
    except Exception as e:
        logger.debug("PaddleOCRVL unavailable: %s", e)

    raise ImportError(
        "Scanned image PDF detected, but PaddleOCR/RapidOCR is not ready in Python.\n"
        "Please run: pip install rapidocr_onnxruntime\n"
        "or: pip install paddleocr"
    )


def _ocr(pdf_path: str | Path, dpi: int = 200) -> tuple[str, str, str, float]:
    """OCR fallback for scanned / image-only PDFs.
    Uses cached pure-Python PaddleOCR / RapidOCR on pypdfium2-rendered page images.
    Returns (full_ocr_text, full_ocr_tables, engine_type, ocr_time_seconds).
    """
    import tempfile
    import time

    start_ocr = time.perf_counter()
    engine, engine_type = _get_ocr_engine()

    ocr_pages: list[str] = []
    ocr_tables: list[str] = []

    scale = dpi / 72  # pypdfium2 uses scale factor (72 DPI base)
    doc = pdfium.PdfDocument(str(pdf_path))
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # Save page image temporarily for OCR engine
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            pil_image.save(tmp_img, format="PNG")
            tmp_img_path = tmp_img.name

        lines_extracted: list[str] = []
        try:
            # Option A: RapidOCR (Fastest ONNX, ~1s per page)
            if engine_type == "rapidocr":
                with open(tmp_img_path, "rb") as f:
                    img_bytes = f.read()
                result, _ = engine(img_bytes)
                if result:
                    for line in result:
                        if line and len(line) > 1:
                            lines_extracted.append(str(line[1]))

            # Option B: Standard PaddleOCR (~2s per page)
            elif engine_type == "paddleocr":
                result = None
                try:
                    result = engine.ocr(tmp_img_path)
                except Exception as pe:
                    logger.warning(
                        "PaddleOCR failed on page %d (%s), falling back to RapidOCR for this page",
                        page_idx + 1, pe,
                    )
                    # Fallback to RapidOCR if Paddle OneDNN crashes on Windows
                    try:
                        from rapidocr_onnxruntime import RapidOCR
                        r_engine = RapidOCR()
                        with open(tmp_img_path, "rb") as f:
                            img_bytes = f.read()
                        r_res, _ = r_engine(img_bytes)
                        if r_res:
                            for line in r_res:
                                if line and len(line) > 1:
                                    lines_extracted.append(str(line[1]))
                    except Exception as e2:
                        logger.warning(
                            "RapidOCR page-level fallback also failed on page %d: %s",
                            page_idx + 1, e2,
                        )

                if result:
                    for page_res in result:
                        if hasattr(page_res, "text") and page_res.text:
                            lines_extracted.append(str(page_res.text))
                        elif isinstance(page_res, dict) and "text" in page_res:
                            lines_extracted.append(str(page_res["text"]))
                        elif isinstance(page_res, (list, tuple)):
                            for line in page_res:
                                if isinstance(line, (list, tuple)) and len(line) > 1 and isinstance(line[1], (list, tuple)) and len(line[1]) > 0:
                                    lines_extracted.append(str(line[1][0]))
                                elif isinstance(line, str):
                                    lines_extracted.append(line)
                        elif hasattr(page_res, "__str__"):
                            lines_extracted.append(str(page_res))

            # Option C: PaddleOCRVL (Heavy VL pipeline)
            elif engine_type == "paddleocr_vl":
                output = engine.predict(tmp_img_path)
                for res in output:
                    if hasattr(res, "markdown") and res.markdown:
                        lines_extracted.append(str(res.markdown))
                    elif hasattr(res, "text") and res.text:
                        lines_extracted.append(str(res.text))
                    elif hasattr(res, "json") and res.json:
                        for item in res.json.get("layout_elements", []):
                            if "text" in item:
                                lines_extracted.append(item["text"])
                    elif isinstance(res, dict) and "text" in res:
                        lines_extracted.append(res["text"])
                    else:
                        lines_extracted.append(str(res))

        finally:
            if os.path.exists(tmp_img_path):
                try:
                    os.unlink(tmp_img_path)
                except Exception:
                    pass  # best-effort temp cleanup, not worth logging

        page.close()

        if lines_extracted:
            ocr_pages.append(f"--- Page {page_idx + 1} (OCR) ---\n" + "\n".join(lines_extracted))

    doc.close()

    ocr_time_seconds = round(time.perf_counter() - start_ocr, 2)
    full_ocr_text = "\n\n".join(ocr_pages).strip()
    full_ocr_tables = "\n\n".join(ocr_tables).strip()
    return full_ocr_text, full_ocr_tables, engine_type, ocr_time_seconds


# A normalize_pdf_rotation() helper used to run here before extract_full_pdf(),
# meant to "fix" a page with /Rotate != 0 by calling pypdf's page.rotate().
# Removed: pypdf's rotate() only overwrites the /Rotate dictionary entry --
# it never transforms the underlying content stream's coordinates. pdfplumber
# already applies /Rotate correctly on its own when reporting word/char
# positions, so zeroing it out post-hoc left pdfplumber reading the SAME raw
# (still-rotated) coordinates as if no rotation were needed, which silently
# scrambled every rotated page's line order. Confirmed on a real 270-degree-
# rotated invoice: calling extract_full_pdf() directly on the untouched file
# produced clean, correctly-ordered text; routing it through this function
# first produced character soup. Simplest correct fix is to not have this
# step at all -- pdfplumber needs no help here.


def _ocr_with_docker(
    input_path: str | Path, dpi: int = 300, force_ocr: bool = False
) -> tuple[Path | None, float]:
    """
    Uses the official OCRmyPDF Docker container (jbarlow83/ocrmypdf-alpine)
    to generate a searchable PDF with exact embedded text coordinates.
    Includes --rotate-pages to automatically detect and correct scanned page orientation using Tesseract OSD.

    By default passes --skip-text, which skips OCR on any page that
    already has *some* text layer -- fast, but it means a page with only
    a sliver of pre-existing garbage text (a stray watermark, a corrupted
    text remnant) gets skipped instead of OCR'd, leaving that page's real
    content unrecovered. `force_ocr=True` passes --force-ocr instead,
    which rasterizes and re-OCRs every page unconditionally -- slower, but
    is the fix when --skip-text is the reason a scan came back empty.

    Returns (Path_to_pdf_or_None, ocr_time_seconds).
    """
    import subprocess
    import tempfile
    import time

    in_file = Path(input_path)
    if not in_file.exists():
        return None, 0.0

    # Fast pre-check: don't attempt (and don't wait out the subprocess
    # timeout for) a Docker call when Docker itself isn't even installed.
    if shutil.which("docker") is None:
        logger.info("Docker OCR fallback skipped: 'docker' not found on PATH")
        return None, 0.0

    start_docker = time.perf_counter()
    tmp_out = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_out.close()

    cmd = [
        "docker", "run", "--rm", "-i",
        "jbarlow83/ocrmypdf-alpine",
        "--image-dpi", str(dpi),
        "--rotate-pages",  # corrects gross 90-degree-multiple orientation
        "--deskew",        # corrects fine-grained tilt (Leptonica-based)
        "--force-ocr" if force_ocr else "--skip-text",
        "-", "-"
    ]

    try:
        with open(in_file, "rb") as infile, open(tmp_out.name, "wb") as outfile:
            proc = subprocess.run(
                cmd,
                stdin=infile,
                stdout=outfile,
                stderr=subprocess.PIPE,
                timeout=180,
            )
        elapsed = round(time.perf_counter() - start_docker, 2)
        if proc.returncode == 0 and os.path.getsize(tmp_out.name) > 0:
            logger.info("Docker OCRmyPDF succeeded in %.2fs", elapsed)
            return Path(tmp_out.name), elapsed
        else:
            logger.info(
                "Docker OCRmyPDF produced no usable output (returncode=%s): %s",
                proc.returncode, proc.stderr.decode(errors="ignore")[:300],
            )
            if os.path.exists(tmp_out.name):
                try:
                    os.unlink(tmp_out.name)
                except Exception:
                    pass
            return None, 0.0
    except subprocess.TimeoutExpired:
        logger.info("Docker OCRmyPDF timed out after 180s, falling back to Python OCR")
        if os.path.exists(tmp_out.name):
            try:
                os.unlink(tmp_out.name)
            except Exception:
                pass
        return None, 0.0
    except Exception as e:
        logger.info("Docker OCR fallback skipped due to error: %s", e)
        if os.path.exists(tmp_out.name):
            try:
                os.unlink(tmp_out.name)
            except Exception:
                pass
        return None, 0.0


def visualize_document_layout(pdf_path: str | Path, page_num: int = 0, dpi: int = 120):
    """
    Visualizes bounding boxes on the PDF page for debugging layout and alignment.
    """
    import matplotlib.pyplot as plt

    pdf_path = Path(pdf_path)
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_num >= len(pdf.pages):
            raise IndexError(f"Page {page_num} out of range (PDF has {len(pdf.pages)} pages).")
        page = pdf.pages[page_num]
        page_img = page.to_image(resolution=dpi)
        words = page.extract_words()
        annotated = page_img.draw_rects(words, stroke="red", stroke_width=1, fill=None)

        fig, ax = plt.subplots(figsize=(10, 10 * (page.height / page.width)))
        ax.imshow(annotated.annotated)
        ax.axis('off')
        ax.set_title(f"Visual Debug: {pdf_path.name} - Page {page_num + 1} ({len(words)} words)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()


def extract_text(
    pdf_path: str | Path,
    min_chars_for_text_layer: int = 40,
    force_ocr: bool | None = None,
) -> tuple[str, str, str, float, dict | None]:
    """
    Returns (text, tables_text, method, ocr_time_seconds, scan_quality).

    - text: High-precision 2D layout canvas preserving exact vertical linearity,
            horizontal column separation, and word collision prevention.
    - tables_text: Empty string (tables are preserved directly within the 2D layout canvas).
    - method: "2d_layout_canvas", "ocrmypdf_docker", or f"ocr_{engine_type}"
    - ocr_time_seconds: time spent running OCR (0.0 for digital PDFs with embedded text layer)
    - scan_quality: scan_validator.validate_pdf()'s result for page 1 (skew_deg,
      whether it's clipped at any edge, valid/reasons), or None if the check
      itself couldn't run. Tilt within a correctable range is already fixed
      in `text` (see extract_full_pdf) -- this is surfaced so callers can
      warn on the cases that AREN'T fixable: severe tilt, or a table border
      that runs off the page edge (content is physically missing, no amount
      of coordinate correction recovers it).
    - force_ocr: controls both whether the digital text-layer path is even
      attempted, and which flag the Docker OCR pass uses:
        * None (default) -- auto. Try the digital text layer first (unless
          is_scanned_pdf() already says this is scanned); if OCR is needed,
          auto-decide --skip-text vs --force-ocr per file based on whether
          any page has a partial text layer (see _has_partial_text_pages()).
        * True  -- always skip the digital-text check and OCR every page,
          using OCRmyPDF's --force-ocr.
        * False -- always try the digital text layer first, and if OCR is
          needed, always use --skip-text (never auto-upgrade to --force-ocr).
    """
    scan_quality: dict | None = None
    if scan_validator is not None:
        try:
            scan_quality = scan_validator.validate_pdf(str(pdf_path), page=0)
            if not scan_quality["valid"]:
                logger.warning("%s: scan quality check failed: %s", pdf_path, scan_quality["reasons"])
        except Exception as e:
            logger.debug("%s: scan quality check could not run (%s)", pdf_path, e)

    skip_digital_path = force_ocr is True

    if not skip_digital_path:
        # 1. Fast path: High-Precision 2D Layout Canvas from embedded text
        # layer -- but only attempt it if is_scanned_pdf() agrees this is a
        # genuine digital PDF. A single sparse page (letterhead, a stamped
        # cover line) could otherwise clear min_chars_for_text_layer on its
        # own while every other page is a pure image.
        already_known_scanned = force_ocr is None and is_scanned_pdf(pdf_path)
        if not already_known_scanned:
            # pdfplumber already applies a page's /Rotate transform itself
            # when reporting word/char coordinates -- no separate
            # normalization step is needed (or safe: see
            # normalize_pdf_rotation's removal note below).
            text = extract_full_pdf(pdf_path)
            if len(text) >= min_chars_for_text_layer:
                return text, "", "2d_layout_canvas", 0.0, scan_quality
            logger.info(
                "Digital text layer in %s was too sparse (<%d chars), falling back to OCR",
                pdf_path, min_chars_for_text_layer,
            )
        else:
            logger.info("%s classified as scanned (sparse/garbage text layer), routing straight to OCR", pdf_path)
    else:
        logger.info("force_ocr=True requested for %s, skipping the digital text-layer check", pdf_path)

    # 2. Decide --skip-text vs --force-ocr for the Docker OCR pass.
    if force_ocr is None:
        use_force_flag = _has_partial_text_pages(pdf_path)
        logger.info(
            "%s: auto-selected %s for the Docker OCR pass",
            pdf_path, "--force-ocr" if use_force_flag else "--skip-text",
        )
    else:
        use_force_flag = force_ocr

    # 3. Scanned PDF / image: Try OCRmyPDF Docker first (with --rotate-pages for auto-orientation)
    docker_pdf, docker_time = _ocr_with_docker(pdf_path, force_ocr=use_force_flag)
    if docker_pdf is not None:
        try:
            d_text = extract_full_pdf(docker_pdf)
            if len(d_text) >= min_chars_for_text_layer:
                return d_text, "", "ocrmypdf_docker", docker_time, scan_quality
        finally:
            if os.path.exists(docker_pdf):
                try:
                    os.unlink(docker_pdf)
                except Exception:
                    pass

    # 4. Fallback to Python OCR (RapidOCR / PaddleOCR / PaddleOCR-VL)
    ocr_text, ocr_tables, engine_type, ocr_time = _ocr(pdf_path)
    return ocr_text, ocr_tables, f"ocr_{engine_type}", ocr_time, scan_quality


if __name__ == "__main__":
    import sys
    t, tabs, m, ot, sq = extract_text(sys.argv[1])
    print(f"--- extracted via {m} ({len(t)} chars, ocr_time: {ot}s) ---")
    if sq:
        print(f"--- scan quality: valid={sq['valid']} skew={sq['skew_deg']} clipped={sq['clipped_edges']} ---")
    print(t[:10000])
