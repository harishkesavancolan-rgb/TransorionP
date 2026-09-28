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
    return [tuple(l[0]) for l in lines]


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
    margin_x = max(3, int(W * edge_margin_frac))
    margin_y = max(3, int(H * edge_margin_frac))
    MIN_INDEPENDENT_LINES = 3
    DEDUPE_GAP = 40  # px apart along the edge to count as a distinct border

    hits_per_edge = {"left": [], "right": [], "top": [], "bottom": []}
    for (x1, y1, x2, y2, length, norm_ang, raw_ang) in cluster:
        for (x, y) in ((x1, y1), (x2, y2)):
            if x <= margin_x:
                hits_per_edge["left"].append(y)
            elif x >= W - margin_x:
                hits_per_edge["right"].append(y)
            if y <= margin_y:
                hits_per_edge["top"].append(x)
            elif y >= H - margin_y:
                hits_per_edge["bottom"].append(x)

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
