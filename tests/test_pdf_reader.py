"""
Tests for pdf_reader.py's _dedupe_overlapping_chars() -- position-based
duplicate-glyph removal used before word/line clustering.
"""
from __future__ import annotations

from unittest.mock import patch

from pdf_reader import (
    _dedupe_overlapping_chars,
    _looks_garbled,
    _word_quality_ratio,
    _maybe_recover_garbled_ocr,
)


def _char(text, top, x0, height=5.16):
    return {"text": text, "top": top, "bottom": top + height, "x0": x0}


def test_tight_space_does_not_eat_the_following_letter():
    # Confirmed real case: a condensed font ("Aptos Narrow, Bold") placed
    # the space before "FROM:" only 0.96pt ahead of the "F" -- tighter
    # than the 1.0pt x-tolerance used to catch genuinely duplicated
    # glyphs. Without the whitespace guard, the space (which sorts first)
    # got kept and the real "F" was flagged as its "duplicate" and
    # dropped, silently eating the first letter of every word on the
    # page ("SHIP FROM:" -> "SHIP ROM:").
    chars = [
        _char("S", top=97.36, x0=19.56),
        _char("H", top=97.36, x0=22.32),
        _char("I", top=97.36, x0=25.69),
        _char("P", top=97.36, x0=27.11),
        _char(" ", top=97.36, x0=29.87),
        _char("F", top=97.36, x0=30.83),  # 0.96pt after the space
        _char("R", top=97.36, x0=33.34),
        _char("O", top=97.36, x0=36.22),
        _char("M", top=97.36, x0=39.58),
    ]
    kept = _dedupe_overlapping_chars(chars)
    assert "".join(c["text"] for c in kept) == "SHIP FROM"


def test_genuine_duplicate_glyph_still_removed():
    # The case this function was originally built for: two overlaid
    # copies of the same visible glyph at slightly different font
    # scales/baselines, decoded via different embedded fonts (so text
    # equality can't be relied on either -- matched by position alone).
    chars = [
        _char("i", top=100.0, x0=50.0, height=5.26),
        _char("i", top=100.4, x0=50.2, height=6.89),  # overlaid duplicate
    ]
    kept = _dedupe_overlapping_chars(chars)
    assert len(kept) == 1


def test_two_distinct_letters_far_enough_apart_both_kept():
    chars = [
        _char("A", top=100.0, x0=50.0),
        _char("B", top=100.0, x0=55.0),
    ]
    kept = _dedupe_overlapping_chars(chars)
    assert len(kept) == 2


# ── _word_quality_ratio / _looks_garbled ────────────────────────────────
# Confirmed real case: an OCRmyPDF page that's actually a clean, high-quality
# scan (just rotated 90 degrees, which --rotate-pages' Tesseract OSD failed
# to detect) comes back as scrambled single-letter fragments -- every word
# shredded because the line-grouping read characters in the wrong order.

_REAL_PROSE_SAMPLE = """
COMMERCIAL INVOICE
Invoice No: 455024201          Issued Date: 14-July-2026
Invoice to: Aptiv Components India Private Ltd.
Unit 1, S.No. 19, SH 48
VARANAVASI VILLAGE & KUNNAVAKKAM VILLAGE
KANCHIPURAM 631604 IN
Item  Description        Part Name      Qty   Unit Price   Total
1     8PGU1592-A          Core Anvil     1     9.5          9.5
"""

_SHREDDED_SAMPLE = """
N                    £
Bed 0
:                 =301
0              W
1 Y
I
]
00A 8
S
V
X
6 0
A
0 1             T7RW1
E
X
d
II              9 1 wnow 0 0
"""


def test_word_quality_ratio_scores_real_prose_highly():
    count, ratio = _word_quality_ratio(_REAL_PROSE_SAMPLE)
    assert count >= 20
    assert ratio > 0.8


def test_word_quality_ratio_scores_shredded_ocr_low():
    count, ratio = _word_quality_ratio(_SHREDDED_SAMPLE)
    assert ratio < 0.5


def test_looks_garbled_false_for_real_prose():
    assert _looks_garbled(_REAL_PROSE_SAMPLE) is False


def test_looks_garbled_true_for_shredded_ocr():
    assert _looks_garbled(_SHREDDED_SAMPLE) is True


def test_looks_garbled_false_when_too_little_text_to_judge():
    # A near-blank page (or a single stamped line) shouldn't trigger an
    # expensive rotation retry just because it has few real words -- there's
    # not enough signal to tell "genuinely sparse" from "scrambled" apart.
    assert _looks_garbled("N d 0 A") is False


# ── _maybe_recover_garbled_ocr ──────────────────────────────────────────

def test_maybe_recover_garbled_ocr_skips_clean_text():
    with patch("pdf_reader._ocr_best_rotation") as mock_recover:
        text, method, ocr_time = _maybe_recover_garbled_ocr(
            "fake.pdf", _REAL_PROSE_SAMPLE, "ocrmypdf_docker", 5.0,
        )
    mock_recover.assert_not_called()
    assert text == _REAL_PROSE_SAMPLE
    assert method == "ocrmypdf_docker"
    assert ocr_time == 5.0


def test_maybe_recover_garbled_ocr_adopts_better_recovery():
    with patch("pdf_reader._ocr_best_rotation", return_value=(_REAL_PROSE_SAMPLE, 0.9, 12.0)):
        text, method, ocr_time = _maybe_recover_garbled_ocr(
            "fake.pdf", _SHREDDED_SAMPLE, "ocrmypdf_docker", 5.0,
        )
    assert text == _REAL_PROSE_SAMPLE
    assert method == "ocrmypdf_docker+rotation_recovered"
    assert ocr_time == 17.0  # original 5.0 + recovery 12.0


def test_maybe_recover_garbled_ocr_keeps_original_when_recovery_no_better():
    # The retry itself came back equally garbled (a genuinely hard-to-OCR
    # document, not a rotation problem) -- don't silently swap in an
    # equally-bad result and hide that this file needs a real fix.
    with patch("pdf_reader._ocr_best_rotation", return_value=(_SHREDDED_SAMPLE, 0.1, 12.0)):
        text, method, ocr_time = _maybe_recover_garbled_ocr(
            "fake.pdf", _SHREDDED_SAMPLE, "ocrmypdf_docker", 5.0,
        )
    assert text == _SHREDDED_SAMPLE
    assert method == "ocrmypdf_docker"
    assert ocr_time == 17.0  # recovery time is still spent/counted even when not adopted
