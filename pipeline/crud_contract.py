"""Single source of truth for the "core CRUD" testability contract every
generated app must follow, parameterized by that app's actual
Requirements (its real field names/types/options and declared filters)
rather than fixed strings — unlike auth_contract.py, which has one fixed
shape, every app's fields differ, so the ids here are derived, not
literal. DeveloperAgent's prompt and QAReviewerAgent's prompt both render
from developer_prompt_block()/qa_prompt_addendum(); browser_tester.py
derives the same ids independently from the same Requirements object, so
the instructions given to the model and the selectors used to grade it
can never drift apart.
"""

from __future__ import annotations

import re

from pipeline.schema import Field, Requirements

ADD_FORM_ID = "add-form"
ADD_SUBMIT_ID = "add-submit"
ITEM_LIST_ID = "item-list"
EDIT_ACTION_SELECTOR = '[data-action="edit"]'
DELETE_ACTION_SELECTOR = '[data-action="delete"]'


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "field"


def field_id(name: str) -> str:
    return f"field-{slugify(name)}"


def filter_id(name: str) -> str:
    return f"filter-{slugify(name)}"


def _find_field(entity, name: str) -> Field | None:
    lowered = name.strip().lower()
    for f in entity.fields:
        if f.name.strip().lower() == lowered:
            return f
    return None


def creatable_fields(requirements: Requirements) -> list[Field]:
    """Primary-entity fields that belong in the add/edit form — every
    field except booleans, which almost never make sense as something you
    set while creating an item (see module docstring on scope)."""
    entity = requirements.primary_entity
    if entity is None:
        return []
    return [f for f in entity.fields if f.type != "boolean"]


def first_testable_filter(requirements: Requirements) -> tuple[str, Field] | None:
    """The first declared filter this project's browser test can drive
    deterministically: a select-typed field with at least two declared
    options, so the test can set it to a value that provably excludes the
    synthetic item and one that provably includes it. Boolean fields are
    excluded here even though the Developer is still asked to make them
    filterable (see developer_prompt_block) — booleans aren't in the
    add-form contract (see creatable_fields), so the test has no way to
    control or know a synthetic item's boolean value, and asserting
    against an unknown default would risk a false failure. Returns
    (filter_name, Field) or None if no declared filter qualifies.
    """
    entity = requirements.primary_entity
    if entity is None:
        return None
    for filter_name in requirements.filters:
        field = _find_field(entity, filter_name)
        if field is not None and field.type == "select" and len(field.options) >= 2:
            return filter_name, field
    return None


def _field_input_line(f: Field) -> str:
    fid = field_id(f.name)
    if f.type == "select":
        options = ", ".join(f.options) if f.options else "options that fit the field"
        return f'  - id="{fid}": a <select> for "{f.name}" with an <option> for each: {options}'
    if f.type == "number":
        return f'  - id="{fid}": a number input for "{f.name}"'
    if f.type == "date":
        return f'  - id="{fid}": a date input for "{f.name}"'
    return f'  - id="{fid}": a text input for "{f.name}"'


def developer_prompt_block(requirements: Requirements) -> str:
    entity = requirements.primary_entity
    if entity is None:
        return ""

    fields = creatable_fields(requirements)
    fields_block = "\n".join(_field_input_line(f) for f in fields) or "  (no non-checkbox fields to include)"

    filter_lines = []
    for filter_name in requirements.filters:
        field = _find_field(entity, filter_name)
        if field is None or field.type not in ("select", "boolean"):
            continue
        fid = filter_id(filter_name)
        if field.type == "boolean":
            filter_lines.append(f'  - id="{fid}": a <select> for filtering by "{filter_name}" with options for "All", "Yes", and "No"')
        else:
            options = ", ".join(field.options) if field.options else "each declared option"
            filter_lines.append(f'  - id="{fid}": a <select> for filtering by "{filter_name}" with an "All" option plus {options}')

    filters_paragraph = ""
    if filter_lines:
        filters_paragraph = (
            f"\nAlso include these live filter controls, each hiding/showing "
            f"#{ITEM_LIST_ID} items when changed:\n" + "\n".join(filter_lines) + "\n"
        )

    return f"""
This app's core create/edit/delete flow must follow a fixed, testable
structure (style and lay it out however fits the app — these ids and
behaviors are exact):
- A form with id="{ADD_FORM_ID}" containing:
{fields_block}
  and a submit control with id="{ADD_SUBMIT_ID}".
- A container with id="{ITEM_LIST_ID}" whose visible text includes every
  item's current field values.
- Each rendered item must contain its own control with
  data-action="edit" and one with data-action="delete".
- Clicking an item's data-action="edit" control must repopulate
  #{ADD_FORM_ID} with that item's current values; submitting
  #{ADD_FORM_ID} afterward must save the change to that SAME item, not
  create a new one.
- Clicking an item's data-action="delete" control must remove it from
  #{ITEM_LIST_ID}.
- #{ADD_FORM_ID} and #{ITEM_LIST_ID} must both be visible without any
  navigation, tab click, or screen switch (logging in first is fine, if
  this app has accounts — but no additional click after that). If your
  app has multiple screens/sections, put the primary entity's management
  UI on the default one shown at load.
{filters_paragraph}
Checkbox/boolean fields on the primary entity (if any) don't belong in
#{ADD_FORM_ID} — expose them however fits the app, e.g. a toggle per item.
"""


def qa_prompt_addendum(requirements: Requirements) -> str:
    entity = requirements.primary_entity
    if entity is None:
        return ""
    declared_filters = ", ".join(requirements.filters) if requirements.filters else "none declared"
    return (
        f'This app must follow the fixed contract: #{ADD_FORM_ID} with one '
        f'id="field-<name>" input per non-checkbox field, id="{ADD_SUBMIT_ID}", '
        f'id="{ITEM_LIST_ID}" showing every item, and each item carrying its own '
        f'data-action="edit"/data-action="delete" controls that genuinely edit/delete '
        f"that specific item (not just visually, and not some other item). Check that "
        f"every declared filter ({declared_filters}) actually filters #{ITEM_LIST_ID}, "
        f'not just whichever one is exposed as a testable id="filter-<name>" select.'
    )
