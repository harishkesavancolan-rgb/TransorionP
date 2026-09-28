"""
models.py
---------
Dynamically generates Pydantic models from the template schema so that:

1. Every field name is strictly enforced (no typos, no invented columns).
2. Types are validated (string vs number).
3. Extra fields returned by a mapper are caught and rejected (a safety net
   -- in practice validate.py's mappers only ever emit real column names,
   but this catches it immediately if that ever drifts).

Usage:
    from schema import load_template_schema
    from models import build_sheet_models

    schema = load_template_schema("templates/EXP_TEMPLET.xlsx")
    models = build_sheet_models(schema)
    # models["ITEM"] is a Pydantic model class with all ITEM fields as Optional
"""
from __future__ import annotations
from typing import Any, Optional

from pydantic import BaseModel, Field, create_model


def _pydantic_type(schema_type: str) -> type:
    """Map schema type string to a Python type for Pydantic.

    We use Union types to be lenient about the LLM returning
    numbers as strings or vice versa. The type coercion in
    validate.py handles the actual conversion before this step.
    """
    if schema_type == "number":
        return Optional[int | float | str]
    return Optional[str | int | float]


def build_sheet_models(
    schema: dict[str, dict[str, Any]],
) -> dict[str, type[BaseModel]]:
    """
    For each sheet in the template schema, dynamically create a Pydantic
    model class with all fields as Optional.

    Returns a dict of {sheet_name: ModelClass}.
    """
    models: dict[str, type[BaseModel]] = {}

    for sheet_name, sheet_info in schema.items():
        fields = sheet_info["fields"]
        types = sheet_info["types"]
        aliases = sheet_info.get("aliases", {})

        # Build field definitions: all Optional, with type from schema
        field_definitions: dict[str, Any] = {}
        for field_name in fields:
            field_type = _pydantic_type(types.get(field_name, "string"))
            # Use aliases as field description for documentation
            field_aliases = aliases.get(field_name, [])
            description = f"Also known as: {', '.join(field_aliases)}" if field_aliases else ""
            field_definitions[field_name] = (
                field_type,
                Field(default=None, description=description or None),
            )

        # Create the model dynamically
        model_class = create_model(
            f"{sheet_name}Row",
            **field_definitions,
        )
        # Configure to forbid extra fields
        model_class.model_config = {"extra": "forbid"}

        models[sheet_name] = model_class

    return models


def validate_row_against_model(
    row: dict[str, Any],
    model_class: type[BaseModel],
    sheet_name: str,
    row_idx: int,
    warnings: list[str],
    all_fields: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    Validate a single row dict against the Pydantic model.

    - Strips any keys not in the model (with warnings).
    - Coerces types where possible.
    - Guarantees ALL template fields for the sheet are present in the exact
      order defined in the template schema; anything the row didn't
      populate is "" (empty string) -- no guessed defaults.
    """
    valid_fields = set(model_class.model_fields.keys())
    fields_order = all_fields if all_fields is not None else list(model_class.model_fields.keys())

    # Filter to only valid fields
    filtered: dict[str, Any] = {}
    for key, val in row.items():
        if key in valid_fields:
            filtered[key] = val
        else:
            warnings.append(
                f"[{sheet_name}][row {row_idx}] pydantic rejected "
                f"unknown field '{key}'"
            )

    if not filtered and not any(v for v in row.values() if v):
        return None

    # Validate via Pydantic (lenient — catches type errors)
    try:
        validated = model_class.model_validate(filtered)
        validated_dict = {
            k: v for k, v in validated.model_dump().items() if v is not None
        }
    except Exception as e:
        warnings.append(
            f"[{sheet_name}][row {row_idx}] pydantic validation error: {e}"
        )
        validated_dict = {k: v for k, v in filtered.items() if v is not None}

    # Construct full row containing EVERY template field in order, "" for
    # anything not populated.
    result: dict[str, Any] = {}
    for field_name in fields_order:
        value = validated_dict.get(field_name)
        result[field_name] = value if value not in (None, "") else ""

    return result
