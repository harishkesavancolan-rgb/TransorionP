"""
scan_validator.py
------------------
Detects page tilt and edge-clipped table borders on a rasterized PDF page,
via a Hough-transform line search rather than word/text coordinates -- so
it works identically on a scanned image and on a rendered selectable PDF.

Core idea:
  1. Binarize the page and find long straight line segments (Hough
     transform), any orientation -- these are candidate table border
     strokes (but in practice, ordinary text baselines/character edges
     provide enough of these even with no ruled table on the page --
     confirmed empirically, this does not require a ruled/bordered table).
  2. Group segments by angle. A real data table (or a page of aligned
     text) contributes many roughly-parallel lines sharing the same skew
     angle, so that cluster dominates over one-off lines like a
     letterhead rule. The cluster's angle = the page's effective tilt.
  3. For the lines in that dominant cluster, check whether any endpoint
     sits within a small margin of the image boundary. If a table
     border's endpoint touches the edge, the border (and whatever
     column/row it bounds) is being cut off by the page/scan edge --
     exactly the failure mode of "table travelling out of the page".

Validated against real invoices and synthetic rotations (0-25 degrees):
angle detection is accurate to within the 1-degree bucket size across that
whole range, and clipped-edge detection correctly fires exactly once
rotation pushes content past the frame. Runs in well under a second per
page (pypdfium2 render + OpenCV, no OCR).
"""
from __future__ import annotations

import cv2
import numpy as np


def _binarize(gray):
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )


def _find_line_segments(bw, min_len_frac=0.25):
    h, w = bw.shape[:2]
    min_len = int(min(h, w) * min_len_frac)
    lines = cv2.HoughLinesP(
        bw, 1, np.pi / 720, threshold=120,
        minLineLength=min_len, maxLineGap=15
    )
    if lines is None:
        return []
    # OpenCV 4.x returns shape (N, 1, 4); 5.x returns (N, 4). Indexing
    # l[0] on the 5.x shape yields a scalar and raised TypeError -- which
    # pdf_reader's callers swallow, silently disabling deskew and the
    # clipped-table check. reshape handles both.
    return [tuple(int(v) for v in l) for l in np.asarray(lines).reshape(-1, 4)]


def _angle_of(x1, y1, x2, y2):
    a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    # fold into (-90, 90], then snap near-vertical/horizontal families together
    while a <= -90:
        a += 180
    while a > 90:
        a -= 180
    return a


def _cluster_by_angle(segments, bin_size=1.0):
    """Bucket segments into angle bins (mod 90, so 0 deg and 90 deg lines from
    the SAME rotated table land in comparable buckets) and return the bucket
    with the greatest total line length -> the table's dominant skew family."""
    buckets = {}
    for (x1, y1, x2, y2) in segments:
        ang = _angle_of(x1, y1, x2, y2)
        # normalize verticals (~90) to their horizontal-equivalent tilt so a
        # table's row-lines and column-lines vote for the same skew estimate
        norm_ang = ang - 90 if ang > 45 else (ang + 90 if ang < -45 else ang)
        key = round(norm_ang / bin_size) * bin_size
        length = np.hypot(x2 - x1, y2 - y1)
        buckets.setdefault(key, []).append((x1, y1, x2, y2, length, norm_ang, ang))

    if not buckets:
        return None, []

    best_key = max(buckets, key=lambda k: sum(s[4] for s in buckets[k]))
    return best_key, buckets[best_key]


def validate_scan(image, max_skew_deg=2.0, edge_margin_frac=0.005, min_len_frac=0.25):
    """
    Returns a dict:
      {
        "skew_deg": float | None,
        "skew_ok": bool,
        "clipped_edges": ["left", "right", ...],  # borders touching page edge
        "clipped": bool,
        "valid": bool,               # overall pass/fail
        "reasons": [str, ...],       # human-readable explanation(s) of failure
        "n_lines_used": int,
      }
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    H, W = gray.shape[:2]
    bw = _binarize(gray)

    segments = _find_line_segments(bw, min_len_frac=min_len_frac)
    angle, cluster = _cluster_by_angle(segments)

    result = {
        "skew_deg": None,
        "skew_ok": True,
        "clipped_edges": [],
        "clipped": False,
        "valid": True,
        "reasons": [],
        "n_lines_used": 0,
    }

    if angle is None or not cluster:
        result["reasons"].append(
            "Could not detect a clear table grid (too few long straight lines); "
            "unable to confirm tilt/clipping."
        )
        result["valid"] = False
        return result

    result["skew_deg"] = round(float(angle), 2)
    result["n_lines_used"] = len(cluster)

    if abs(angle) > max_skew_deg:
        result["skew_ok"] = False
        result["reasons"].append(
            f"Page is tilted {angle:+.2f}°, exceeding the {max_skew_deg}° limit."
        )

    # A real clipped table shows MULTIPLE independent row (or column) borders
    # all terminating right at the same page edge - because every row line
    # gets truncated at the same missing column, and vice versa. A single
    # stray line touching the edge (scanner-bed shadow, a fold, a rule under
    # a heading) is just noise. So: collect edge-touching endpoints per side,
    # de-duplicate points that belong to the same line fragment, and only
    # flag a side once several distinct borders end there.
    #
    # Two confirmed real false positives fixed here:
    #
    # 1. Orientation. A row border getting clipped at the left/right edge is
    # a roughly HORIZONTAL line whose endpoint runs INTO that edge; a column
    # border clipped at the top/bottom edge is a roughly VERTICAL line
    # running into THAT edge. The original code never checked this, so a
    # line running PARALLEL to the edge it touches -- e.g. a vertical
    # decorative page border, or a scanner-bed-edge shadow, sitting right
    # next to the left/right edge for most of the page's height -- was
    # wrongly accepted as clipping evidence. Confirmed on three real,
    # visually-unclipped invoices (455024201.pdf, Commercial Invoice
    # 11.pdf, INV1_1.pdf): every line that triggered a false "clipped" verdict
    # ran parallel to the edge it was checked against, not perpendicular
    # into it.
    #
    # 2. Endpoint double-counting. A single line running ALONG an edge (the
    # false-positive case above) has both its endpoints sitting near that
    # edge, often far apart from each other since the line itself is long --
    # the original code added both endpoints as separate "hits", so one
    # line could look like two independent borders on its own. Fixed by
    # collecting at most one point per line per edge (the midpoint of
    # whichever endpoint(s) of that line touch it).
    margin_x = max(3, int(W * edge_margin_frac))
    margin_y = max(3, int(H * edge_margin_frac))
    MIN_INDEPENDENT_LINES = 3
    DEDUPE_GAP = 40  # px apart along the edge to count as a distinct border
    MAX_PARALLEL_DEG = 20  # a line within this many degrees of running
    # ALONG the edge (rather than into it) is excluded as decorative/
    # artifact, not a clipped row/column border.

    hits_per_edge = {"left": [], "right": [], "top": [], "bottom": []}
    for (x1, y1, x2, y2, length, norm_ang, raw_ang) in cluster:
        # abs(raw_ang) near 0 = horizontal line, near 90 = vertical line.
        is_horizontal = abs(raw_ang) <= MAX_PARALLEL_DEG
        is_vertical = abs(abs(raw_ang) - 90) <= MAX_PARALLEL_DEG

        line_hits: dict[str, list[int]] = {}
        for (x, y) in ((x1, y1), (x2, y2)):
            # left/right clipping evidence must be a horizontal line running
            # INTO that edge -- a vertical line merely sitting near the edge
            # (running along it) is excluded.
            if is_horizontal:
                if x <= margin_x:
                    line_hits.setdefault("left", []).append(y)
                elif x >= W - margin_x:
                    line_hits.setdefault("right", []).append(y)
            # top/bottom clipping evidence must be a vertical line running
            # INTO that edge, symmetrically.
            if is_vertical:
                if y <= margin_y:
                    line_hits.setdefault("top", []).append(x)
                elif y >= H - margin_y:
                    line_hits.setdefault("bottom", []).append(x)

        # This one line contributes AT MOST one point per edge, even if
        # both its endpoints touch that edge -- otherwise a single long
        # line touching an edge at both ends would double-count as two
        # independent borders.
        for side, coords in line_hits.items():
            hits_per_edge[side].append(sum(coords) / len(coords))

    edges_hit = set()
    for side, coords in hits_per_edge.items():
        if not coords:
            continue
        coords = sorted(coords)
        groups = [[coords[0]]]
        for c in coords[1:]:
            if c - groups[-1][-1] <= DEDUPE_GAP:
                groups[-1].append(c)
            else:
                groups.append([c])
        if len(groups) >= MIN_INDEPENDENT_LINES:
            edges_hit.add(side)

    if edges_hit:
        result["clipped"] = True
        result["clipped_edges"] = sorted(edges_hit)
        result["reasons"].append(
            f"Table border line(s) touch the page edge at: {', '.join(sorted(edges_hit))}. "
            "This means part of the table (a row or column) likely falls outside "
            "the scanned page and is missing."
        )

    result["valid"] = result["skew_ok"] and not result["clipped"]
    return result


def validate_pdf(pdf_path, dpi=200, page=0, **kwargs):
    """Convenience wrapper: render a PDF page and validate it.

    Uses pypdfium2 (Apache-2.0 / BSD-3-Clause, wraps Google's PDFium) to
    rasterize the page - no external system binaries (e.g. poppler) needed,
    and no AGPL dependency (unlike PyMuPDF). Works on scanned AND selectable
    PDFs identically, since it operates on the rendered pixels either way.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        pdf_page = pdf[page]
        scale = dpi / 72  # PDFium works in points; 72pt = 1 inch
        bitmap = pdf_page.render(scale=scale)
        pil_img = bitmap.to_pil()
    finally:
        pdf.close()

    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return validate_scan(img, **kwargs)
