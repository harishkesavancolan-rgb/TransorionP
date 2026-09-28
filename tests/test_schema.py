"""Smoke tests for schema.py against the real templates in templates/."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import load_template_schema

_TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def test_export_template_has_item_sheet():
    schema = load_template_schema(_TEMPLATES / "EXP_TEMPLET.xlsx")
    assert "ITEM" in schema
    assert "Item_Ser_No" in schema["ITEM"]["fields"]
    assert "Item_Desc" in schema["ITEM"]["fields"]


def test_import_template_has_boe_not_item():
    schema = load_template_schema(_TEMPLATES / "IMP_TEMPLET.xlsx")
    assert "BOE" in schema
    assert "ITEM" not in schema
    assert "SL_No" in schema["BOE"]["fields"]


def test_known_manual_aliases_resolve():
    schema = load_template_schema(_TEMPLATES / "EXP_TEMPLET.xlsx")
    aliases = schema["ITEM"]["aliases"]
    assert "HSN Code" in aliases["Item_RITC"]
    assert "Country of Origin" in aliases["Itm_Source_Cntry"]


def test_field_types_inferred():
    schema = load_template_schema(_TEMPLATES / "EXP_TEMPLET.xlsx")
    types = schema["ITEM"]["types"]
    assert types["Item_Qty"] == "number"
    assert types["Item_Desc"] == "string"
    assert types["Printed"] == "string"  # forced string despite no number hint anyway
