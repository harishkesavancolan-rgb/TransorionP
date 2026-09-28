"""
templates_config.py
--------------------
Single source of truth for "which template/sheet applies to which
shipment type" -- previously this {"export": ..., "import": ...} mapping
was copy-pasted independently in main.py, api.py, and validate.py, which
is exactly the kind of duplication this project is otherwise trying to
avoid. Everything that needs it now imports from here instead.
"""
from __future__ import annotations
from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

TEMPLATE_PATH: dict[str, Path] = {
    "export": TEMPLATES_DIR / "EXP_TEMPLET.xlsx",
    "import": TEMPLATES_DIR / "IMP_TEMPLET.xlsx",
}

# The sheet that holds line items in each template -- export's EXP_TEMPLET.xlsx
# calls it "ITEM"; import's IMP_TEMPLET.xlsx has no "ITEM" sheet at all and
# calls its line-item sheet "BOE" (Bill of Entry) instead.
ITEM_SHEET_NAME: dict[str, str] = {"export": "ITEM", "import": "BOE"}
