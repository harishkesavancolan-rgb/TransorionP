"""
detect_type.py
--------------
Decides whether an invoice is an IMPORT or EXPORT shipment, purely from the
extracted PDF text -- before any LLM call and before a template is chosen.

Why this has to run first: the export template (EXP_TEMPLET.xlsx, line
items on the "ITEM" sheet) and the import template (IMP_TEMPLET.xlsx, line
items on the "BOE" sheet) have almost no columns in common. Something has
to pick the right one before extraction can even start, and it has to be
cheap and deterministic -- not a second paid LLM call -- since it's pure
gating logic, not a field to report to the user.

Scoring is weighted, not "any keyword present": multi-word customs phrases
are unambiguous, but bare 3-letter acronyms (BCD, CVD, IGM...) collide with
unrelated text often enough that they're only worth a token vote, not a
deciding one.

If neither side clearly wins, the result is "unknown" -- this module never
guesses. Callers (main.py / batch.py / api.py) are expected to require an
explicit --type/`type=` override in that case rather than pick one anyway.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ShipmentType = Literal["import", "export", "unknown"]


class AmbiguousShipmentTypeError(RuntimeError):
    """Raised when auto-detection can't confidently call import vs. export.

    Callers should catch this distinctly from other extraction failures --
    it means "ask the user/caller to pass --type explicitly", not "this
    file is broken".
    """

# ── Keyword signals ─────────────────────────────────────────────────────
# Strong: unambiguous multi-word customs phrases. Weak: acronyms that can
# collide with unrelated text (part numbers, random 3-letter codes, etc.)
# so they only get a small vote.

_EXPORT_STRONG = (
    "shipping bill", "let export order", "export invoice", "drawback",
    "export tax invoice", "expwp", "expwop",
)
_EXPORT_WEAK = ("rodtep", "meis", "egm")
# "expwp"/"expwop" are India's GST e-invoice "Supply Type" codes for
# Export With Payment / Without Payment of tax -- confirmed real case:
# an invoice (Nokia Solutions and Networks India, IRN present) had no
# label matching _SUPPLIER_LABELS anywhere (its own address block used
# first-person framing, "Our Registered Office Address", not "Exporter:"/
# "Seller:"), so the only structural signal available was this GST
# Supply Type field ("GST   EXPWP   N   INV"). As formal, government-
# defined GST e-invoice codes -- not a loose acronym that could collide
# with unrelated text -- these are as unambiguous as "shipping bill", so
# they're STRONG, not WEAK. Only meaningful on export: these are OUTWARD-
# supply codes an Indian GST-registered seller puts on their own invoice;
# a foreign supplier's commercial invoice on a genuine import has no
# reason to carry one.

_IMPORT_STRONG = (
    "bill of entry", "boe no", "into bond", "warehouse bond",
    "import invoice", "customs duty payable", "import tax invoice",
)
_IMPORT_WEAK = ("igm", "bcd", "cvd", "swc")

_STRONG_POINTS = 3
_WEAK_POINTS = 1
# Originally lower than _STRONG_POINTS on the assumption structural signals
# were corroboration only, never sufficient alone. Real invoices tested
# against this module invalidated that: commercial invoices routinely have
# ZERO keyword hits (no "bill of entry"/"shipping bill" phrasing at all --
# that's customs-filing vocabulary, not invoice vocabulary), so a
# structural signal is very often the ONLY signal available. Since
# _origin_destination_signal() and the buyer/supplier proximity check are
# both already conservative (return no signal rather than guess when
# ambiguous), a hit from either is treated as equally strong evidence as
# an unambiguous keyword phrase, not lesser corroboration -- matching
# _STRONG_POINTS lets a single clean structural hit clear _DECISION_MARGIN
# on its own instead of being stuck just short of it.
_STRUCTURAL_POINTS = _STRONG_POINTS
_DECISION_MARGIN = 3  # export_score - import_score must clear this to decide

# Must match pdf_reader.py's _COLUMN_BOUNDARY_MARKER -- this module treats
# it as pure formatting (like whitespace) when judging whether two values
# sit adjacent to each other, same as pdf_reader.py treats it as a column
# separator rather than real content.
_COLUMN_BOUNDARY_MARKER = "│"

# Origin/destination labels -- broadened from the original exact-phrase
# match after real invoices showed variants like "Country of Origin of
# Gds" and "Exporter Country" / "Country of Final Dest." that the
# original `country\s+of\s+origin`-only pattern never matched at all.
# "country of loading (of goods)" and "country of export" both added
# after confirmed real invoices used those exact phrases (not "port of
# loading"/"port of export", already covered) as their ONLY origin-side
# label -- with no origin label recognized at all, the code fell through
# to reading just the destination label's own nearest country and got
# the reading backwards (see _origin_destination_signal's docstring on
# the "NL" case for a related failure in that same fallback). Confirmed
# on a real IMPORT invoice specifically: "Country of Export: Malaysia │
# Country of Destination: India" read as an EXPORT (origin=India) before
# this, the exact reverse of the truth, because only the destination
# label was recognized.
_ORIGIN_LABEL_RE = re.compile(
    r"country\s+of\s+origin(?:\s+of\s+\w+)?|exporter\s+country|"
    r"country\s+of\s+loading(?:\s+of\s+\w+)?|country\s+of\s+export|"
    r"port\s+of\s+loading|port\s+of\s+export",
    re.IGNORECASE,
)
_DEST_LABEL_RE = re.compile(
    r"country\s+of\s+(?:final\s+)?dest(?:ination)?\.?|port\s+of\s+discharge|"
    r"final\s+destination|port\s+of\s+import",
    re.IGNORECASE,
)

# Common country names that show up as origin/destination VALUES.
# Deliberately not exhaustive -- just enough to pick "this token is a
# country name" out from surrounding label/noise text (see
# _origin_destination_signal for why that's the actual problem here,
# not just matching the label).
_COUNTRY_NAMES = (
    "india", "usa", "united states", "uk", "united kingdom", "germany",
    "singapore", "china", "maldives", "malaysia", "japan", "france",
    "italy", "spain", "netherlands", "belgium", "canada", "australia",
    "thailand", "vietnam", "indonesia", "korea", "taiwan", "mexico",
    "brazil", "switzerland", "uae", "saudi arabia", "hong kong",
    "south africa", "nigeria", "kenya", "egypt", "turkey", "russia",
    "poland", "sweden", "norway", "denmark", "finland", "austria",
    "portugal", "ireland", "greece", "israel", "qatar", "kuwait",
    "oman", "bangladesh", "sri lanka", "nepal", "pakistan", "myanmar",
    "philippines", "new zealand", "argentina", "chile", "colombia",
    "peru", "ukraine", "czech republic", "romania", "hungary",
)
_COUNTRY_NAME_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _COUNTRY_NAMES) + r")\b",
    re.IGNORECASE,
)

# Descriptive prefixes that precede a country's formal name but aren't
# themselves in _COUNTRY_NAMES -- see _fallback_pairing_is_reliable for
# why this matters (a gap ending in one of these should still count as
# "no real interrupting content").
_COUNTRY_NAME_PREFIX_RE = re.compile(
    r"(republic\s+of|kingdom\s+of|state\s+of|sultanate\s+of|principality\s+of|"
    r"united\s+states\s+of|people'?s\s+republic\s+of)\s*$",
    re.IGNORECASE,
)

# Second structural signal: most real commercial invoices skip explicit
# "Country of Origin/Destination" labels entirely, but reliably have a
# buyer/consignee address block and a supplier/exporter address block.
# If exactly one side's label is immediately followed by "India" (the
# other side isn't), that's a strong directional signal even with zero
# keyword hits -- e.g. "Receiver: ABB India Limited ... INDIA" on a
# waybill, or "Customer ... ABB INDIA LIMITED" on a commercial invoice.
_BUYER_LABELS = ("customer", "buyer", "consignee", "bill to", "ship to",
                 "sold to", "invoice address", "importer", "receiver")
_SUPPLIER_LABELS = ("supplier", "seller", "exporter", "shipper", "sold by",
                    "vendor", "manufacturer", "ship from")
# "ship from" added after a confirmed real invoice had it as the ONLY
# label naming the actual shipping-origin address (an Indian factory) --
# the invoice's own "Exporter" label named a different, US-based billing
# entity instead, so the supplier-side proximity check found no India
# there and missed the signal entirely; "ship from" is the natural,
# already-covered counterpart to "ship to" (in _BUYER_LABELS below).
_LABEL_PROXIMITY_WINDOW = 400  # chars scanned after the label for "India"
# Was 200: a real invoice ("SOLD-TO" -> "...Agilent Technologies India...")
# had "india" starting at character 198 after the label -- so close that
# only 2 of its 5 characters fell inside a 200-char window and the whole
# match silently missed. VAT ID numbers, tax codes, and multi-line
# addresses routinely push the actual country name further from its
# label than 200 chars covers.
_LABEL_PROXIMITY_POINTS = _STRONG_POINTS

# Search window for _origin_destination_signal's single-label fallback
# (only origin OR only destination is labeled -- see that function's
# docstring for why this must be bounded, not unbounded to end-of-text).
# Deliberately smaller than _LABEL_PROXIMITY_WINDOW: a country name right
# next to its own label is typically within the same cell/line, unlike
# "India" appearing anywhere in a multi-line address block after a
# buyer/supplier label.
_SINGLE_LABEL_SEARCH_WINDOW = 150


# A bare label like "buyer"/"supplier" also matches inside a REFERENCE-
# NUMBER field's own name -- "Buyer's Order No. & Date", "Buyer's
# Reference", "Supplier's Invoice No." -- which names the OTHER party's
# paperwork, not this section's own address block. Confirmed on a real
# invoice: "BUYER'S ORDER NO. & DATE" sits in one column while the
# EXPORTER's own address wraps onto the next line of a DIFFERENT column;
# once pdf_reader.py's 2D canvas flattens both into one text stream, the
# exporter's own "...CHENNAI...INDIA." ends up textually right after
# "BUYER'S ORDER NO." by coincidence of column layout -- a proximity scan
# anchored there wrongly read it as "the buyer is in India" on a genuine
# EXPORT invoice with an Indian exporter (confirmed: 6059-FWD INV.pdf,
# INV - RGP-005 -V1.pdf both hit this exact pattern and came back
# "unknown" from a real, resolvable signal getting silently reversed).
# The possessive is optionally apostrophe-less too -- "Buyers Order No /
# Date" (no apostrophe) is a confirmed real variant of "Buyer's Order
# No." that `(?:'s)?` alone didn't match (it requires the apostrophe),
# letting this exact same false-match class back in under that spelling.
_REFERENCE_FIELD_RE = re.compile(
    r"^\s*(?:'?s)?\s*(order|ref\w*|po\b|invoice|date)\b", re.IGNORECASE,
)


def _is_reference_field_label(text_lower: str, match_end: int) -> bool:
    return bool(_REFERENCE_FIELD_RE.match(text_lower[match_end:match_end + 20]))


# A run of 2+ letters, used to tell "genuinely nothing before this match in
# its own cell" from "this sits after some other real word."
_WORD_RE = re.compile(r"[a-z]{2,}")


def _starts_its_own_cell(text_lower: str, match_start: int) -> bool:
    """
    True if `match_start` sits at the start of its own column cell (the
    text since the line's own start or the nearest preceding "│",
    whichever is closer) -- i.e. this reads like a genuine field label,
    not a label-vocabulary word sitting mid-phrase inside a DIFFERENT
    field's value.

    Confirmed real false positive without this: "seller" matched inside
    "FCA Seller' Premises" -- an Incoterms phrase that is itself the
    VALUE of an unrelated "DELIVERY TERMS:" field, not a party label at
    all -- and a proximity scan anchored there read across a further "│"
    into a THIRD, unrelated column's "...India..." on the same row,
    wrongly registering a supplier-side India hit on a genuine IMPORT
    invoice. A real field label always starts its own cell (only
    whitespace/punctuation before it, no other word); "seller" here had
    "FCA " in front of it within the same cell, which this catches.
    """
    line_start = text_lower.rfind("\n", 0, match_start) + 1
    col_start = text_lower.rfind(_COLUMN_BOUNDARY_MARKER, line_start, match_start)
    cell_start = col_start + 1 if col_start != -1 else line_start
    return not _WORD_RE.search(text_lower[cell_start:match_start])


def _label_positions(text_lower: str, labels: tuple[str, ...]) -> list[tuple[int, int]]:
    """(start, end) for every match of every label in `labels`, excluding
    reference-field false matches (see _is_reference_field_label) and
    matches that aren't at the start of their own column cell (see
    _starts_its_own_cell)."""
    positions = []
    for label in labels:
        pattern = r"[\s\-]+".join(re.escape(w) for w in label.split())
        for m in re.finditer(pattern, text_lower):
            if not _is_reference_field_label(text_lower, m.end()) and _starts_its_own_cell(text_lower, m.start()):
                positions.append((m.start(), m.end()))
    return positions


def _label_proximity_hits(
    text_lower: str,
    labels: tuple[str, ...],
    other_side_starts: list[int] = (),
) -> int:
    """Counts labels (buyer- or supplier-side) that have "India" within
    `_LABEL_PROXIMITY_WINDOW` characters after them. Each label counts at
    most once, so a label repeated many times (e.g. "India" printed twice
    in one address block) doesn't dominate the score.

    Multi-word labels match with `[\\s-]+` between words, not just `\\s+`:
    a real invoice printed "SOLD-TO" / "SHIP-TO" (hyphenated, no
    whitespace at all) for exactly the fields this is meant to catch --
    a whitespace-only separator silently never matched those at all.

    The window is also capped at the start of the next `other_side_starts`
    position, if one falls inside it: a real waybill had its Shipper and
    Receiver blocks sitting right next to each other, close enough that a
    plain 400-char window starting at "Shipper" reached straight into the
    Receiver block's own "India" -- a supplier-side window has no business
    reading into the very next party's address, no matter how far away
    _LABEL_PROXIMITY_WINDOW allows.

    An "India" that's the object of "to" -- "Ship From: Changzhou,CN To
    India" -- is excluded (see _has_own_india): that's a route
    description naming India as the DESTINATION, not evidence that the
    "Ship From" party's own location is India. Confirmed on a real
    invoice: exactly that phrase, on a genuine IMPORT (China -> India)
    invoice, made "ship from" (added to _SUPPLIER_LABELS for a different,
    genuine case) wrongly register a supplier/exporter-in-India hit.
    """
    hits = 0
    for label in labels:
        pattern = r"[\s\-]+".join(re.escape(w) for w in label.split())
        for m in re.finditer(pattern, text_lower):
            if _is_reference_field_label(text_lower, m.end()):
                continue
            if not _starts_its_own_cell(text_lower, m.start()):
                continue
            window_end = m.end() + _LABEL_PROXIMITY_WINDOW
            next_other = min((p for p in other_side_starts if p > m.end()), default=None)
            if next_other is not None:
                window_end = min(window_end, next_other)
            window = _same_column_window(text_lower, m.end(), window_end)
            if _has_own_india(window):
                hits += 1
                break
    return hits


def _same_column_window(text: str, start: int, end: int) -> str:
    """
    Returns text[start:end], but with every line AFTER the label's own
    line restricted to the SAME column slot (as split by
    `_COLUMN_BOUNDARY_MARKER`) the label itself sits in -- so a
    multi-line address block never reads into a DIFFERENT party's block
    that happens to sit column-adjacent to it a few rows down.

    The label's own (first) line is deliberately left unrestricted: "│"
    is also routinely used WITHIN a single row as a plain label/value
    separator for several unrelated fields packed side by side (e.g.
    "Ship To Messrs: │ Lenovo India Pvt Ltd. │ FCR#", or "Country of
    Origin │ : India") -- confirmed real regression from an earlier,
    stricter version of this function that also restricted the first
    line: it cut "Ship To Messrs:" off from its own value sitting right
    after the very next "│", silently dropping a correct India hit and
    turning a previously-working detection into "unknown". That
    label-then-value-then-next-field pattern is common and needs the full
    first line, unrestricted, to keep working.

    Confirmed real false positive this DOES still catch (on later lines):
    a supplier's own UK address (left column: "SELLER/ SUPPLIER:" /
    "HYVE SOLUTIONS EUROPE LIMITED:" / ...) sat row-by-row next to the
    buyer's own address (right column, ending in "...ZIP CODE 700054" /
    "INDIA") -- two unrelated parties' blocks rendered side by side. The
    false India hit sat on the SECOND row of that block, not the label's
    own first row, so restricting only continuation lines still excludes
    it while leaving the label's own line alone.

    A line with no "│" at all is taken whole -- this restriction only
    matters on rows that actually have sibling-column content to exclude;
    a normal multi-line address block within a single column is
    unaffected.
    """
    line_start = text.rfind("\n", 0, start) + 1
    col_idx = text.count(_COLUMN_BOUNDARY_MARKER, line_start, start)

    parts: list[str] = []
    pos = start
    first = True
    while pos < end:
        nl = text.find("\n", pos, end)
        line_end = nl if nl != -1 else end
        line = text[pos:line_end]
        if first:
            parts.append(line)
        elif _COLUMN_BOUNDARY_MARKER in line:
            segments = line.split(_COLUMN_BOUNDARY_MARKER)
            parts.append(segments[col_idx] if col_idx < len(segments) else "")
        else:
            parts.append(line)
        first = False
        if nl == -1:
            break
        pos = line_end + 1
    return "\n".join(parts)


_DESTINATION_PREFIX_RE = re.compile(r"\bto\s*$", re.IGNORECASE)


def _has_own_india(window: str) -> bool:
    """True if "india" appears in `window` other than as the destination
    of a "... To India" route phrase (see _label_proximity_hits)."""
    for m in re.finditer(r"\bindia\b", window):
        prefix = window[max(0, m.start() - 6):m.start()]
        if not _DESTINATION_PREFIX_RE.search(prefix):
            return True
    return False


def _fallback_pairing_is_reliable(gap: str) -> bool:
    """
    Decides whether the reading-order fallback in _origin_destination_signal
    can be trusted for this document, based on `gap` -- the raw text
    sitting BETWEEN the two candidate country values themselves (not
    between the labels and the first value: an earlier version measured
    that instead, and it broke on a real multi-column table where two
    OTHER columns' content -- "Pre Carriage by" / "Place of Receipt by
    Pre-Carrier" -- legitimately sits between the label row and the
    origin/destination columns on the very same, correctly-aligned value
    row; that's harmless sibling-column text, not an interrupting block,
    and flagging it as unreliable wrongly threw away a correct pairing).

    Measuring the gap between the two VALUES instead sidesteps that. But
    "no newline allowed" turned out to be too strict on its own: one
    otherwise-correct real invoice pairs its values on two CONSECUTIVE
    lines (origin on one line, destination directly below it) -- gap is
    just "\n", still perfectly reliable, but a bare newline-count check
    would reject it. The actual signal isn't newlines at all, it's
    CONTENT: is there any real text between the two values, or only
    formatting (whitespace, the "│" column marker)? On every confirmed-
    correct case (DR REDD'S x2, both Oceanic patterns) the gap strips down
    to nothing once whitespace and "│" are removed. On the broken
    VIBRACOUSTIC case, real words remain ("Tesla, Inc.", "NA-US-TX-
    Kyle-201...") -- that's genuine interrupting content, not formatting.
    """
    residue = gap.strip().replace(_COLUMN_BOUNDARY_MARKER, "").strip()
    # A formal country name is often prefixed by a descriptive word that
    # _COUNTRY_NAME_RE's bare-country-word matching doesn't capture --
    # "Republic of Korea", "Kingdom of Saudi Arabia" match only on
    # "korea"/"saudi arabia", so the prefix is left sitting in the GAP
    # between the two values instead of being recognized as the start of
    # the second value's own name. Confirmed on a real invoice: the gap
    # was "... | Republic of " immediately before "Korea" -- genuine
    # formatting plus the beginning of "Republic of Korea" itself, not
    # interrupting content, but it was reading as unreliable (residue
    # "Republic of") and silently discarding an otherwise-correct
    # origin/destination pairing.
    residue = _COUNTRY_NAME_PREFIX_RE.sub("", residue).strip()
    return residue == ""


def _origin_destination_signal(text: str) -> tuple[bool, bool]:
    """
    Returns (origin_is_india, destination_is_india) -- best-effort.

    The hard part here was never matching the label (a regex handles
    that fine); it's that the VALUE frequently isn't adjacent to its own
    label at all. Real examples: "Country of Origin of Gds Country of
    Final Dest." on one line, then "INDIA          |          Maldives"
    on the NEXT line -- the values are column-paired with each other via
    a wide gap, not textually close to either label. Worse: naively
    picking "whichever label is textually closest to the word India"
    gets this specific case backwards, since "Country of Final Dest."
    sits textually closer to "INDIA" than "Country of Origin" does, even
    though India is the origin value here (they're aligned by column,
    not by proximity).

    So instead: find each label's own position, then take the first
    recognized country name (see _COUNTRY_NAMES) appearing after THAT
    label specifically, bounded so it can't run past the OTHER label's
    position and grab that label's value instead. Resolving origin and
    destination independently like this (rather than assuming "origin
    label always prints before destination label, so the first two
    country names in reading order are origin-then-destination in that
    order") matters because that assumption doesn't always hold: e.g.
    some invoices print "CTRY/PORT OF IMPORT: France" BEFORE "CTRY/PORT
    OF EXPORT: India" -- taking the first two countries after whichever
    label starts earliest would silently swap origin and destination.
    Deliberately returns (False, False) for a side with no signal --
    no guessing -- rather than pick a country that isn't actually this
    label's own value. This includes the single-label fallback below
    (only one of origin/destination is labeled at all): its search window
    is bounded, not unbounded to end-of-document -- confirmed on a real
    export invoice whose "Country of Destination" value was abbreviated
    "NL" (not in _COUNTRY_NAMES), so an unbounded scan skipped right past
    it and grabbed an unrelated "India" mention much further down the
    page, misreading the destination as India on a shipment that was
    actually going TO the Netherlands. Bounding the window turns that
    into "no country found nearby" (no signal) instead of a confidently
    wrong answer -- consistent with this function's whole design.
    """
    origin_match = _ORIGIN_LABEL_RE.search(text)
    dest_match = _DEST_LABEL_RE.search(text)
    if not origin_match and not dest_match:
        return False, False

    def _nearest_country(after: int, bound: int | None) -> str | None:
        limit = len(text) if bound is None else min(len(text), bound)
        m = _COUNTRY_NAME_RE.search(text, after, limit) if limit > after else None
        return m.group(1).lower() if m else None

    origin_country = None
    dest_country = None

    if origin_match and dest_match:
        origin_bound = dest_match.start() if dest_match.start() > origin_match.end() else None
        dest_bound = origin_match.start() if origin_match.start() > dest_match.end() else None
        origin_country = _nearest_country(origin_match.end(), origin_bound)
        dest_country = _nearest_country(dest_match.end(), dest_bound)

        if origin_country is None or dest_country is None:
            # Direct per-label lookup found nothing for at least one side --
            # typically because the two labels sit back-to-back on one line
            # ("...Origin of Gds Country of Final Dest.") with their actual
            # values column-paired on a LATER line, not adjacent to either
            # label's own text. Fall back to reading order: take the first
            # two country names after the earlier label, assigned to
            # whichever label comes first/second spatially (not "origin is
            # always first" -- some invoices print the destination label
            # first, e.g. "CTRY/PORT OF IMPORT" before "CTRY/PORT OF
            # EXPORT").
            start = min(origin_match.start(), dest_match.start())
            country_matches = list(_COUNTRY_NAME_RE.finditer(text[start:]))
            if len(country_matches) >= 2:
                # Reading order only matches label order when the two
                # candidate values themselves sit right next to each other
                # (see _fallback_pairing_is_reliable). Confirmed real
                # counter-example: a "DELIVERY ADDRESS & NOTIFY:" block
                # interleaved between the origin/destination labels and
                # their own values (a taller side-column sharing the same
                # rows) pushed the destination's value onto a different
                # row than the origin's, silently swapping them if the
                # pairing were trusted blindly.
                gap = text[start + country_matches[0].end():start + country_matches[1].start()]
                if _fallback_pairing_is_reliable(gap):
                    found = [m.group(1).lower() for m in country_matches[:2]]
                    if origin_match.start() <= dest_match.start():
                        origin_country, dest_country = found[0], found[1]
                    else:
                        dest_country, origin_country = found[0], found[1]
    elif origin_match:
        origin_country = _nearest_country(origin_match.end(), origin_match.end() + _SINGLE_LABEL_SEARCH_WINDOW)
    else:
        dest_country = _nearest_country(dest_match.end(), dest_match.end() + _SINGLE_LABEL_SEARCH_WINDOW)

    return origin_country == "india", dest_country == "india"


def _phrase_present(text_lower: str, phrase: str) -> bool:
    """
    True if `phrase` appears in `text_lower`. Multi-word phrases match with
    flexible whitespace (`\\s+` between words, not a literal single space):
    pdf_reader.py's 2D layout canvas preserves column gaps as runs of many
    spaces, so "EXPORT      INVOICE" (title spaced out across a column) is
    common and a literal `"export invoice" in text_lower` check would miss
    it entirely.
    """
    if " " in phrase:
        pattern = r"\s+".join(re.escape(w) for w in phrase.split())
        return re.search(pattern, text_lower) is not None
    return re.search(rf"\b{re.escape(phrase)}\b", text_lower) is not None


def _count(text_lower: str, phrases: tuple[str, ...], weak: bool) -> int:
    points = _WEAK_POINTS if weak else _STRONG_POINTS
    return sum(points for phrase in phrases if _phrase_present(text_lower, phrase))


@dataclass
class DetectionResult:
    shipment_type: ShipmentType
    export_score: int
    import_score: int
    signal: Literal["keyword", "structural", "both", "none"] = "none"
    matched: list[str] = field(default_factory=list)


def detect_shipment_type(text: str) -> DetectionResult:
    """Classify raw invoice text as "import", "export", or "unknown"."""
    text_lower = text.lower()
    matched: list[str] = []

    export_score = _count(text_lower, _EXPORT_STRONG, weak=False)
    export_score += _count(text_lower, _EXPORT_WEAK, weak=True)
    import_score = _count(text_lower, _IMPORT_STRONG, weak=False)
    import_score += _count(text_lower, _IMPORT_WEAK, weak=True)

    keyword_hit = export_score > 0 or import_score > 0
    for phrase in (*_EXPORT_STRONG, *_EXPORT_WEAK):
        if _phrase_present(text_lower, phrase):
            matched.append(f"export:{phrase}")
    for phrase in (*_IMPORT_STRONG, *_IMPORT_WEAK):
        if _phrase_present(text_lower, phrase):
            matched.append(f"import:{phrase}")

    # Structural corroboration: origin/destination relative to India.
    structural_hit = False
    origin_is_india, dest_is_india = _origin_destination_signal(text)
    if origin_is_india and not dest_is_india:
        export_score += _STRUCTURAL_POINTS
        structural_hit = True
        matched.append("structural:origin=IN,destination=foreign")
    elif dest_is_india and not origin_is_india:
        import_score += _STRUCTURAL_POINTS
        structural_hit = True
        matched.append("structural:destination=IN,origin=foreign")

    buyer_starts = [start for start, _ in _label_positions(text_lower, _BUYER_LABELS)]
    supplier_starts = [start for start, _ in _label_positions(text_lower, _SUPPLIER_LABELS)]
    buyer_india = _label_proximity_hits(text_lower, _BUYER_LABELS, other_side_starts=supplier_starts)
    supplier_india = _label_proximity_hits(text_lower, _SUPPLIER_LABELS, other_side_starts=buyer_starts)
    if buyer_india and not supplier_india:
        import_score += _LABEL_PROXIMITY_POINTS
        structural_hit = True
        matched.append("structural:buyer/consignee address is in India")
    elif supplier_india and not buyer_india:
        export_score += _LABEL_PROXIMITY_POINTS
        structural_hit = True
        matched.append("structural:supplier/exporter address is in India")

    if keyword_hit and structural_hit:
        signal: Literal["keyword", "structural", "both", "none"] = "both"
    elif keyword_hit:
        signal = "keyword"
    elif structural_hit:
        signal = "structural"
    else:
        signal = "none"

    diff = export_score - import_score
    if diff >= _DECISION_MARGIN:
        shipment_type: ShipmentType = "export"
    elif -diff >= _DECISION_MARGIN:
        shipment_type = "import"
    else:
        shipment_type = "unknown"

    return DetectionResult(
        shipment_type=shipment_type,
        export_score=export_score,
        import_score=import_score,
        signal=signal,
        matched=matched,
    )


if __name__ == "__main__":
    import sys
    sample = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore") if len(sys.argv) > 1 else ""
    print(detect_shipment_type(sample))
