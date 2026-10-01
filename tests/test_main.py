"""
Tests for main.py's pre-extraction ClippedTableError gate -- confirms a
confirmed-clipped scan is refused BEFORE extract_text() (and therefore
before any OCR or LLM cost) runs, unless explicitly overridden.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from main import run
from pdf_reader import ClippedTableError


class _StopAfterGate(Exception):
    """Sentinel raised by a mocked extract_text() so a test can prove the
    gate let execution continue past it, without needing to mock the rest
    of the (expensive, LLM-calling) pipeline."""


def _run(pdf_path="fake.pdf", **kwargs):
    return asyncio.run(run(pdf_path, "gpt-5-nano", "out.json", **kwargs))


def test_clipped_scan_raises_before_extract_text():
    with patch("main.check_scan_quality", return_value={"clipped": True, "clipped_edges": ["left"], "valid": False, "reasons": ["x"]}), \
         patch("main.extract_text") as mock_extract_text:
        with pytest.raises(ClippedTableError):
            _run()
    mock_extract_text.assert_not_called()


def test_clipped_scan_allowed_with_override():
    with patch("main.check_scan_quality", return_value={"clipped": True, "clipped_edges": ["left"], "valid": False, "reasons": ["x"]}), \
         patch("main.extract_text", side_effect=_StopAfterGate):
        with pytest.raises(_StopAfterGate):
            _run(allow_clipped=True)


def test_unclipped_scan_proceeds_past_gate():
    with patch("main.check_scan_quality", return_value={"clipped": False, "clipped_edges": [], "valid": True, "reasons": []}), \
         patch("main.extract_text", side_effect=_StopAfterGate):
        with pytest.raises(_StopAfterGate):
            _run()


def test_no_scan_quality_result_proceeds_past_gate():
    # scan_validator unavailable / check failed -- treated as "couldn't
    # determine", not "confirmed clipped", so extraction still proceeds.
    with patch("main.check_scan_quality", return_value=None), \
         patch("main.extract_text", side_effect=_StopAfterGate):
        with pytest.raises(_StopAfterGate):
            _run()
