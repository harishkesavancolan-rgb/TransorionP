"""
Tests for pdf_reader.py's _dedupe_overlapping_chars() -- position-based
duplicate-glyph removal used before word/line clustering.
"""
from __future__ import annotations

from pdf_reader import _dedupe_overlapping_chars


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
