"""Drives a generated HTML app in a real headless browser to prove its
core functionality actually works: login (when the app has accounts,
pipeline/auth_contract.py's fixed ids) followed by the primary entity's
add/edit/delete/filter flow (pipeline/crud_contract.py's per-app derived
ids). When the app has accounts, auth runs first and ends logged in,
since a plausible app design gates the entity UI behind login — CRUD is
then driven on that same, already-authenticated page.

Playwright + a Chromium binary are a dev/test-only dependency (see
requirements-dev.txt) — this module tolerates either being missing and
degrades to "not executed" rather than failing the pipeline, since a bare
production deployment (e.g. Streamlit Community Cloud) won't have a
browser available.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pipeline import crud_contract
from pipeline.auth_contract import (
    AUTH_STATUS_ID,
    LOGIN_PASSWORD_ID,
    LOGIN_SUBMIT_ID,
    LOGIN_USERNAME_ID,
    REGISTER_PASSWORD_ID,
    REGISTER_SUBMIT_ID,
    REGISTER_USERNAME_ID,
)
from pipeline.schema import Field, Requirements, TestReport

ACTION_TIMEOUT_MS = 5000
SYNTHETIC_USERNAME = "qa_synth_user"
SYNTHETIC_PASSWORD = "Str0ngP@ss!"
# Doubles as both the "wrong password" login attempt and the password used
# to try re-registering the same username — see _drive_auth_phase.
SECOND_PASSWORD = "totally-wrong-password"


def run_browser_functional_test(requirements: Requirements, has_auth: bool, html: str) -> TestReport:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        return TestReport(
            passed=True,
            executed=False,
            notes=["Playwright isn't installed in this environment; the testing stage was skipped."],
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        html_path = Path(tmp_dir) / "app.html"
        html_path.write_text(html, encoding="utf-8")

        try:
            with sync_playwright() as p:
                launch_kwargs = {"headless": True, "args": ["--no-sandbox"]}
                executable_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
                if executable_path:
                    launch_kwargs["executable_path"] = executable_path
                try:
                    browser = p.chromium.launch(**launch_kwargs)
                except PlaywrightError as exc:
                    return TestReport(
                        passed=True,
                        executed=False,
                        notes=[f"No browser binary available in this environment ({exc}); the testing stage was skipped."],
                    )
                try:
                    return _drive_functional_flow(browser, html_path, requirements, has_auth)
                finally:
                    browser.close()
        except Exception as exc:  # the generated app misbehaving shouldn't crash the pipeline
            return TestReport(passed=False, executed=True, notes=[f"Browser test crashed: {exc}"])


def _drive_functional_flow(browser, html_path: Path, requirements: Requirements, has_auth: bool) -> TestReport:
    page = browser.new_page()
    page.set_default_timeout(ACTION_TIMEOUT_MS)
    page.route("**/*", lambda route: route.continue_() if route.request.url.startswith("file://") else route.abort())

    url = html_path.as_uri()
    page.goto(url)

    notes: list[str] = []
    passed = True

    if has_auth:
        auth_passed, auth_notes = _drive_auth_phase(page, url)
        passed = passed and auth_passed
        notes.extend(auth_notes)

    crud_passed, crud_notes = _drive_crud_phase(page, requirements)
    passed = passed and crud_passed
    notes.extend(crud_notes)

    return TestReport(passed=passed, executed=True, notes=notes)


def _drive_auth_phase(page, url: str) -> tuple[bool, list[str]]:
    if page.locator(f"#{REGISTER_USERNAME_ID}").count() == 0:
        return False, [f"Missing #{REGISTER_USERNAME_ID} — cannot test registration/login."]

    notes: list[str] = []

    page.fill(f"#{REGISTER_USERNAME_ID}", SYNTHETIC_USERNAME)
    page.fill(f"#{REGISTER_PASSWORD_ID}", SYNTHETIC_PASSWORD)
    page.click(f"#{REGISTER_SUBMIT_ID}")
    page.wait_for_timeout(300)

    # Try to re-register the same username with a different password. If
    # duplicate rejection is broken, this silently becomes the account's
    # new password — which the wrong/correct-password checks below catch
    # for free, since SECOND_PASSWORD would then succeed instead of fail.
    page.fill(f"#{REGISTER_USERNAME_ID}", SYNTHETIC_USERNAME)
    page.fill(f"#{REGISTER_PASSWORD_ID}", SECOND_PASSWORD)
    page.click(f"#{REGISTER_SUBMIT_ID}")
    page.wait_for_timeout(300)

    # Reload to prove accounts persist via localStorage, not an in-page value.
    page.goto(url)

    page.fill(f"#{LOGIN_USERNAME_ID}", SYNTHETIC_USERNAME)
    page.fill(f"#{LOGIN_PASSWORD_ID}", SECOND_PASSWORD)
    page.click(f"#{LOGIN_SUBMIT_ID}")
    page.wait_for_timeout(300)
    passed = True
    if SYNTHETIC_USERNAME in _auth_status_text(page):
        passed = False
        notes.append(
            "Logging in with the password from a duplicate-registration attempt succeeded — "
            "duplicate usernames must be rejected, not silently overwrite the existing account."
        )
    else:
        notes.append("Correctly rejected both a wrong password and a duplicate-registration attempt.")

    page.fill(f"#{LOGIN_USERNAME_ID}", SYNTHETIC_USERNAME)
    page.fill(f"#{LOGIN_PASSWORD_ID}", SYNTHETIC_PASSWORD)
    page.click(f"#{LOGIN_SUBMIT_ID}")
    page.wait_for_timeout(300)
    status_after_correct = _auth_status_text(page)
    if SYNTHETIC_USERNAME not in status_after_correct:
        passed = False
        notes.append(
            f"After registering and reloading, logging in with the original password did not reflect a "
            f"logged-in state in #{AUTH_STATUS_ID} (got {status_after_correct!r})."
        )
    else:
        notes.append("Registered a synthetic user, reloaded, and logged in successfully afterward.")

    return passed, notes


def _auth_status_text(page) -> str:
    locator = page.locator(f"#{AUTH_STATUS_ID}")
    return locator.inner_text() if locator.count() else ""


def _drive_crud_phase(page, requirements: Requirements) -> tuple[bool, list[str]]:
    if page.locator(f"#{crud_contract.ADD_FORM_ID}").count() == 0:
        return False, [f"Missing #{crud_contract.ADD_FORM_ID} — cannot test add/edit/delete."]

    fields = crud_contract.creatable_fields(requirements)
    if not fields:
        return True, ["No non-checkbox fields on the primary entity to test creation with; add/edit/delete wasn't executed."]

    notes: list[str] = []
    passed = True
    marker_field = next((f.name for f in fields if f.type == "text"), fields[0].name)

    original_values = _synthetic_values(fields, variant=1)
    updated_values = _synthetic_values(fields, variant=2)

    _fill_fields(page, fields, original_values)
    page.click(f"#{crud_contract.ADD_SUBMIT_ID}")
    page.wait_for_timeout(300)

    marker = original_values[marker_field]
    if marker not in _item_list_text(page):
        return False, [f"After adding an item, {marker!r} did not appear in #{crud_contract.ITEM_LIST_ID}."]
    notes.append(f"Added an item and saw it appear in #{crud_contract.ITEM_LIST_ID}.")

    current_values = original_values
    edit_control = page.locator(f"#{crud_contract.ITEM_LIST_ID} {crud_contract.EDIT_ACTION_SELECTOR}").first
    if edit_control.count() == 0:
        passed = False
        notes.append(f"No {crud_contract.EDIT_ACTION_SELECTOR} control found inside #{crud_contract.ITEM_LIST_ID} — cannot test editing.")
    else:
        edit_control.click()
        page.wait_for_timeout(300)
        _fill_fields(page, fields, updated_values)
        page.click(f"#{crud_contract.ADD_SUBMIT_ID}")
        page.wait_for_timeout(300)
        list_text = _item_list_text(page)
        new_marker = updated_values[marker_field]
        if new_marker not in list_text:
            passed = False
            notes.append(f"After editing, {new_marker!r} did not appear in #{crud_contract.ITEM_LIST_ID}.")
        elif marker in list_text:
            passed = False
            notes.append(
                f"After editing, the original value {marker!r} was still present alongside the new one — "
                "this may have created a duplicate instead of updating the item in place."
            )
        else:
            notes.append("Edited the item in place and saw the updated value replace the original.")
        current_values = updated_values
        marker = new_marker

    filter_check = crud_contract.first_testable_filter(requirements)
    if filter_check is None:
        notes.append("No declared filter targets a testable select field; filter behavior wasn't executed (left to Code Review).")
    else:
        filter_name, field = filter_check
        if field.name not in current_values:
            notes.append(f"Filter {filter_name!r} targets a field outside the add-form contract; filter behavior wasn't executed.")
        else:
            filter_passed, filter_notes = _drive_filter_check(page, filter_name, field, current_values[field.name])
            passed = passed and filter_passed
            notes.extend(filter_notes)

    delete_control = page.locator(f"#{crud_contract.ITEM_LIST_ID} {crud_contract.DELETE_ACTION_SELECTOR}").first
    if delete_control.count() == 0:
        passed = False
        notes.append(f"No {crud_contract.DELETE_ACTION_SELECTOR} control found inside #{crud_contract.ITEM_LIST_ID} — cannot test deleting.")
    else:
        delete_control.click()
        page.wait_for_timeout(300)
        if marker in _item_list_text(page):
            passed = False
            notes.append(f"After deleting, {marker!r} was still present in #{crud_contract.ITEM_LIST_ID}.")
        else:
            notes.append("Deleted the item and it no longer appears in the list.")

    return passed, notes


def _drive_filter_check(page, filter_name: str, field: Field, item_value: str) -> tuple[bool, list[str]]:
    fid = crud_contract.filter_id(filter_name)
    if page.locator(f"#{fid}").count() == 0:
        return False, [f"Missing #{fid} for the declared filter {filter_name!r} — cannot test filtering."]

    excluding_option = next((opt for opt in field.options if opt != item_value), None)
    if excluding_option is None:
        return True, [f"Filter {filter_name!r} has no alternate option to test exclusion with; skipped."]

    notes: list[str] = []
    passed = True

    page.select_option(f"#{fid}", excluding_option)
    page.wait_for_timeout(300)
    if item_value in _item_list_text(page):
        passed = False
        notes.append(f"Setting #{fid} to {excluding_option!r} did not hide the item (still showing {item_value!r}).")
    else:
        notes.append(f"Filter {filter_name!r} correctly hid the item when set to a non-matching value.")

    page.select_option(f"#{fid}", item_value)
    page.wait_for_timeout(300)
    if item_value not in _item_list_text(page):
        passed = False
        notes.append(f"Setting #{fid} back to {item_value!r} did not show the item again.")
    else:
        notes.append(f"Filter {filter_name!r} correctly showed the item again when reset to a matching value.")

    return passed, notes


def _synthetic_values(fields: list[Field], variant: int) -> dict[str, str]:
    values: dict[str, str] = {}
    for f in fields:
        if f.type == "select":
            if f.options:
                idx = 0 if variant == 1 else min(1, len(f.options) - 1)
                values[f.name] = f.options[idx]
            else:
                values[f.name] = f"Option {variant}"
        elif f.type == "number":
            values[f.name] = "1" if variant == 1 else "2"
        elif f.type == "date":
            values[f.name] = "2026-01-01" if variant == 1 else "2026-02-02"
        else:
            values[f.name] = f"Synthetic {f.name} {variant}"
    return values


def _fill_fields(page, fields: list[Field], values: dict[str, str]) -> None:
    for f in fields:
        selector = f"#{crud_contract.field_id(f.name)}"
        if f.type == "select":
            page.select_option(selector, values[f.name])
        else:
            page.fill(selector, values[f.name])


def _item_list_text(page) -> str:
    locator = page.locator(f"#{crud_contract.ITEM_LIST_ID}")
    return locator.inner_text() if locator.count() else ""
