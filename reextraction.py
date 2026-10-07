"""
reextraction.py
-----------------
Orchestrates: extract -> validate -> (if invalid) targeted re-extract
with feedback -> validate again, up to a configurable retry limit.

Both main.py and api.py call the single function here
(extract_and_validate) instead of each duplicating this control flow --
this module owns the retry POLICY; llm_extract.py (call the model),
validate.py (map generic JSON onto template columns), and
invoice_validator.py (check consistency) each stay focused on their own
single responsibility, unchanged in how they work standalone.

Retry policy (spec): MAX_REEXTRACTION_ATTEMPTS = 2 retries AFTER the
initial extraction -- 3 attempts total. Only ERROR-level, retryable
validation findings (invoice_validator.RETRYABLE_RULES) trigger a retry;
WARNINGs never do. Re-extraction is TARGETED: if only the header pass's
output failed validation, only the header pass is re-run on retry (the
line-items pass's already-valid result is reused as-is), and vice versa
-- see build_retry_feedback() in invoice_validator.py, which returns
separate feedback strings per pass for exactly this reason.
"""
from __future__ import annotations

import logging
from typing import Any

from llm_extract import (
    extract_invoice_header,
    extract_line_items_chunked as extract_line_items,
    apply_header_guardrails,
    apply_line_item_guardrails,
    _calculate_cost,
    _to_float,
)
from validate import validate_and_coerce
from invoice_validator import validate_invoice, build_retry_feedback, ValidationResult

logger = logging.getLogger("invoice_extractor")


def _line_sum_gap(items: list[dict[str, Any]], printed_total: float) -> float | None:
    """How far the items' line_totals are from the printed invoice total, or
    None if any item has no numeric line_total (the sum then can't be
    compared to a whole-invoice figure)."""
    # Rows with no quantity, price or amount at all are stray fragments (they
    # are discarded later in validate.py) and don't take part in the sum.
    rows = [
        it for it in items
        if isinstance(it, dict) and any(
            _to_float(it.get(k)) is not None for k in ("quantity", "unit_price", "line_total")
        )
    ]
    amounts = [_to_float(it.get("line_total")) for it in rows]
    if not amounts or any(a is None for a in amounts):
        return None
    return abs(printed_total - sum(amounts))


def _is_closer_to_total(new_items: list[dict[str, Any]], old_items: list[dict[str, Any]],
                        header_raw: dict[str, Any]) -> bool:
    """For the "rows went missing" retry: keep the new item list only if its
    line amounts are at least as close to the printed total as the previous
    list's. Without a single numeric printed total there's nothing to judge
    by, so the new list is accepted."""
    total = _to_float(header_raw.get("invoice_total"))
    if total is None:
        return True
    new_gap, old_gap = _line_sum_gap(new_items, total), _line_sum_gap(old_items, total)
    if old_gap is None:
        return True
    return new_gap is not None and new_gap <= old_gap

# "2 retries after the initial extraction" per the spec -- 3 attempts
# total. Configurable per-call via extract_and_validate's max_retries
# parameter; this is only the default.
MAX_REEXTRACTION_ATTEMPTS = 1


async def extract_and_validate(
    invoice_text: str,
    schema: dict[str, Any],
    shipment_type: str,
    model: str = "gpt-5-nano",
    tables_text: str = "",
    api_key: str | None = None,
    max_retries: int = MAX_REEXTRACTION_ATTEMPTS,
) -> dict[str, Any]:
    """
    Drop-in replacement for the old two-line
    `raw = await extract_fields(...); sheets, warnings, header =
    validate_and_coerce(raw, schema, shipment_type)` sequence that used
    to live directly in main.py's run() and api.py's _process_pdf()
    (extract_fields() itself has since been removed -- it was a
    convenience wrapper nothing in production called anymore once this
    function took over) -- same inputs, and the return dict still has
    `header`, `sheets`, `warnings`, `token_usage` with the same meaning
    as before, so existing callers only need to add handling for the new
    keys rather than rewrite anything.

    New keys on the returned dict:
      validation          -- final invoice_validator.ValidationResult.to_dict()
      validation_history  -- [{"attempt": N, "valid": bool, "errors": [...]}, ...]
                              one entry per attempt made, in order
      attempts            -- how many attempts were made (1 to max_retries+1)
      extraction_status   -- "validated" | "validation_failed"
    """
    header_raw: dict[str, Any] = {}
    items_raw: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_usage = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    validation_history: list[dict[str, Any]] = []
    # Notes from the deterministic guardrails, surfaced in the output's
    # warnings. Kept per pass and reset only when THAT pass is re-extracted,
    # so a note never outlives the model output it describes.
    header_notes: list[str] = []
    item_notes: list[str] = []

    header_feedback: str | None = None
    items_feedback: str | None = None
    # "Rows went missing" (LINE_ITEMS_INCOMPLETE) gets ONE special retry: the
    # items pass re-run one page per call (see extract_line_items_chunked's
    # fine_chunks). If that doesn't close the gap, it isn't repeated.
    fine_next = False
    fine_tried = False
    attempt = 0
    result: dict[str, Any] = {"header": {}, "sheets": {}}
    validation: ValidationResult | None = None

    while True:
        attempt += 1
        # On attempt 1 both passes always run. On a retry, only the
        # pass(es) whose output actually failed validation are re-run --
        # build_retry_feedback() leaves the other one as None.
        need_header = attempt == 1 or header_feedback is not None
        need_items = attempt == 1 or items_feedback is not None

        if need_header:
            header_raw, h_usage = await extract_invoice_header(
                invoice_text, tables_text=tables_text, model=model, api_key=api_key,
                feedback=header_feedback, previous_result=header_raw,
            )
            for k in total_usage:
                total_usage[k] += h_usage.get(k, 0)

        if need_items:
            previous_items = items_raw
            extra = {"fine_chunks": True} if fine_next else {}
            new_items, i_usage = await extract_line_items(
                invoice_text, tables_text=tables_text, model=model, api_key=api_key,
                feedback=items_feedback, previous_result=previous_items, **extra,
            )
            for k in total_usage:
                total_usage[k] += i_usage.get(k, 0)
            # Repair the model's output from the page BEFORE judging it: a
            # complete list whose amounts include a wrong-column value (which
            # the repair fixes) must not lose to a shorter list just because
            # its RAW amounts sit further from the printed total.
            new_item_notes = apply_line_item_guardrails(new_items, invoice_text)
            # Applies to EVERY items re-extraction, not just the completeness
            # retry: on a real Flowserve invoice, attempt 2 recovered all 15
            # rows with the exact printed total (one quantity misread), then
            # a feedback retry for that quantity came back with 13 rows 1,020
            # short -- and that worse list became the final answer.
            if attempt > 1 and not _is_closer_to_total(new_items, previous_items, header_raw):
                logger.warning(
                    "Items re-extraction did not bring the line items closer to the printed "
                    "total -- keeping the previous item list."
                )
                # items_raw and item_notes stay with the previous list
            else:
                items_raw, item_notes = new_items, new_item_notes
            fine_next = False

        if need_header:
            header_notes = []
        if need_header or need_items:
            new_notes: list[str] = []
            apply_header_guardrails(
                header_raw, items_raw, invoice_text, shipment_type=shipment_type, notes=new_notes,
            )
            header_notes.extend(n for n in new_notes if n not in header_notes)

        raw ={"invoice_header": header_raw, "line_items": items_raw}
        sheets, warnings, header = validate_and_coerce(raw, schema, shipment_type)
        warnings = header_notes + item_notes + warnings
        result = {"header": header, "sheets": sheets}

        validation = validate_invoice(result, invoice_text, shipment_type)
        validation_history.append({
            "attempt": attempt,
            "valid": validation.valid,
            "errors": [c.to_dict() for c in validation.errors],
        })

        if validation.valid or attempt > max_retries:
            break

        fb_validation = validation
        if fine_tried:
            # The page-by-page completeness retry has already been spent;
            # don't ask for it again, but other retryable errors still count.
            fb_validation = ValidationResult(
                valid=validation.valid, status=validation.status,
                checks=[c for c in validation.checks if c.rule != "LINE_ITEMS_INCOMPLETE"],
            )
        header_feedback, items_feedback = build_retry_feedback(fb_validation)
        fine_next = not fine_tried and any(
            c.rule == "LINE_ITEMS_INCOMPLETE" for c in validation.retryable_errors
        )
        if fine_next:
            fine_tried = True
        if header_feedback is None and items_feedback is None:
            # Every ERROR present was non-retryable (its rule isn't in
            # RETRYABLE_RULES) -- there's nothing a re-extraction attempt
            # could act on, so stop instead of looping pointlessly.
            break

        logger.warning(
            "Validation failed on attempt %d (%d error(s)) -- re-extracting "
            "(%s).", attempt, len(validation.errors),
            "header+items" if header_feedback and items_feedback
            else "header only" if header_feedback else "items only",
        )

    token_usage = _calculate_cost(
        model, total_usage["input_tokens"], total_usage["cached_tokens"], total_usage["output_tokens"]
    )

    return {
        "header": result["header"],
        "sheets": result["sheets"],
        "warnings": warnings,
        "token_usage": token_usage,
        "validation": validation.to_dict(),
        "validation_history": validation_history,
        "attempts": attempt,
        "extraction_status": "validated" if validation.valid else "validation_failed",
    }
