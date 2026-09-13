"""Drives a generated HTML app in a real headless browser to prove its
registration/login flow actually works, using the fixed element-id
contract in pipeline/auth_contract.py.

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

from pipeline.auth_contract import (
    AUTH_STATUS_ID,
    LOGIN_PASSWORD_ID,
    LOGIN_SUBMIT_ID,
    LOGIN_USERNAME_ID,
    REGISTER_PASSWORD_ID,
    REGISTER_SUBMIT_ID,
    REGISTER_USERNAME_ID,
)
from pipeline.schema import TestReport

ACTION_TIMEOUT_MS = 5000
SYNTHETIC_USERNAME = "qa_synth_user"
SYNTHETIC_PASSWORD = "Str0ngP@ss!"
WRONG_PASSWORD = "totally-wrong-password"


def run_browser_auth_test(html: str) -> TestReport:
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
                    return _drive_auth_flow(browser, html_path)
                finally:
                    browser.close()
        except Exception as exc:  # the generated app misbehaving shouldn't crash the pipeline
            return TestReport(passed=False, executed=True, notes=[f"Browser test crashed: {exc}"])


def _drive_auth_flow(browser, html_path: Path) -> TestReport:
    notes: list[str] = []
    passed = True

    page = browser.new_page()
    page.set_default_timeout(ACTION_TIMEOUT_MS)
    page.route("**/*", lambda route: route.continue_() if route.request.url.startswith("file://") else route.abort())

    url = html_path.as_uri()
    page.goto(url)

    if page.locator(f"#{REGISTER_USERNAME_ID}").count() == 0:
        return TestReport(
            passed=False,
            executed=True,
            notes=[f"Missing #{REGISTER_USERNAME_ID} — cannot test registration/login."],
        )

    page.fill(f"#{REGISTER_USERNAME_ID}", SYNTHETIC_USERNAME)
    page.fill(f"#{REGISTER_PASSWORD_ID}", SYNTHETIC_PASSWORD)
    page.click(f"#{REGISTER_SUBMIT_ID}")
    page.wait_for_timeout(300)

    # Reload to prove accounts persist (the actual point of localStorage
    # over an in-page variable) rather than just surviving within one load.
    page.goto(url)

    page.fill(f"#{LOGIN_USERNAME_ID}", SYNTHETIC_USERNAME)
    page.fill(f"#{LOGIN_PASSWORD_ID}", WRONG_PASSWORD)
    page.click(f"#{LOGIN_SUBMIT_ID}")
    page.wait_for_timeout(300)
    status_after_wrong = _auth_status_text(page)
    if SYNTHETIC_USERNAME in status_after_wrong:
        passed = False
        notes.append("Login succeeded with a wrong password — should have been rejected.")
    else:
        notes.append("Correctly rejected a login attempt with the wrong password.")

    page.fill(f"#{LOGIN_USERNAME_ID}", SYNTHETIC_USERNAME)
    page.fill(f"#{LOGIN_PASSWORD_ID}", SYNTHETIC_PASSWORD)
    page.click(f"#{LOGIN_SUBMIT_ID}")
    page.wait_for_timeout(300)
    status_after_correct = _auth_status_text(page)
    if SYNTHETIC_USERNAME not in status_after_correct:
        passed = False
        notes.append(
            f"After registering and reloading, logging in with the correct password did not reflect a "
            f"logged-in state in #{AUTH_STATUS_ID} (got {status_after_correct!r})."
        )
    else:
        notes.append("Registered a synthetic user, reloaded the page, and logged in successfully afterward.")

    return TestReport(passed=passed, executed=True, notes=notes)


def _auth_status_text(page) -> str:
    locator = page.locator(f"#{AUTH_STATUS_ID}")
    return locator.inner_text() if locator.count() else ""
