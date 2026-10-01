"""
Tests for scan_validator.py's clipped-table-edge detection.

Confirmed real bug (found by manually inspecting 3 real invoices that were
being wrongly refused as "clipped"): the original code counted a hit
whenever a line's endpoint sat near a page edge, with no check on the
line's ORIENTATION relative to that edge, and added BOTH of a line's
endpoints as separate hits. A single long line running ALONG an edge (a
decorative page border, or a scanner-bed-edge shadow) -- not evidence of a
clipped table at all -- would then look like two-or-more "independent
borders" on its own. These tests build synthetic images with drawn lines
to lock in the fix: a genuine clipped row/column border (perpendicular
into the edge) must still be caught, and a line merely running alongside
an edge (parallel to it) must not.
"""
from __future__ import annotations

import numpy as np
import cv2

import scan_validator as sv

_W, _H = 600, 800


def _blank_image():
    return np.full((_H, _W, 3), 255, dtype=np.uint8)


def _draw_line(img, x1, y1, x2, y2, thickness=2):
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)


def test_line_running_along_left_edge_is_not_clipping():
    # Two vertical line fragments both sitting right at the left edge,
    # running along it for most of the page height -- exactly the shape of
    # a decorative outer page border or a scanner-edge shadow. Confirmed
    # real false positive: this pattern wrongly flagged 3 real, visually
    # uncut invoices before the orientation check was added.
    img = _blank_image()
    _draw_line(img, 2, 50, 2, 400)
    _draw_line(img, 2, 450, 2, 750)
    result = sv.validate_scan(img)
    assert result["clipped"] is False
    assert "left" not in result["clipped_edges"]


def test_horizontal_lines_terminating_at_left_edge_is_clipping():
    # Three independent horizontal lines at different heights, each
    # genuinely cut off at the left edge -- the real signature of a table
    # column that's fallen off the scanned page (every row's own border
    # truncated at the same missing column).
    img = _blank_image()
    _draw_line(img, 2, 100, 300, 100)
    _draw_line(img, 2, 300, 300, 300)
    _draw_line(img, 2, 500, 300, 500)
    result = sv.validate_scan(img)
    assert result["clipped"] is True
    assert "left" in result["clipped_edges"]


def test_line_running_along_top_edge_is_not_clipping():
    # Symmetric case for the top edge: horizontal lines running ALONG it
    # (e.g. a letterhead rule or barcode boundary sitting near the top)
    # must not count as clipped columns.
    img = _blank_image()
    _draw_line(img, 50, 2, 250, 2)
    _draw_line(img, 300, 2, 550, 2)
    result = sv.validate_scan(img)
    assert result["clipped"] is False
    assert "top" not in result["clipped_edges"]


def test_vertical_lines_terminating_at_top_edge_is_clipping():
    # Three independent vertical lines each genuinely cut off at the top
    # edge -- the real signature of a clipped row.
    img = _blank_image()
    _draw_line(img, 100, 2, 100, 300)
    _draw_line(img, 300, 2, 300, 300)
    _draw_line(img, 500, 2, 500, 300)
    result = sv.validate_scan(img)
    assert result["clipped"] is True
    assert "top" in result["clipped_edges"]


def test_single_line_does_not_meet_independent_line_minimum():
    # Even a genuinely perpendicular line shouldn't be enough alone --
    # MIN_INDEPENDENT_LINES requires several before concluding a real
    # column/row is missing, not just one stray mark near the edge.
    img = _blank_image()
    _draw_line(img, 2, 300, 300, 300)
    result = sv.validate_scan(img)
    assert result["clipped"] is False
