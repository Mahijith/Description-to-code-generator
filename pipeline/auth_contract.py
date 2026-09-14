"""Single source of truth for the "app has accounts" testability contract.

When ArchitectAgent decides an app needs registration/login, DeveloperAgent
is instructed to wire up exactly these element ids, and browser_tester.py
drives those same ids with a headless browser. Keeping both sides here
means the prompt text and the Playwright selectors can never drift apart.
"""

from __future__ import annotations

REGISTER_USERNAME_ID = "register-username"
REGISTER_PASSWORD_ID = "register-password"
REGISTER_SUBMIT_ID = "register-submit"
LOGIN_USERNAME_ID = "login-username"
LOGIN_PASSWORD_ID = "login-password"
LOGIN_SUBMIT_ID = "login-submit"
AUTH_STATUS_ID = "auth-status"

DEVELOPER_PROMPT_BLOCK = f"""
This app needs basic user accounts. Store accounts in `localStorage`
(under a key of your choosing, separate from the app's main data) so
registered users are still there next time the file is opened — this is a
small embedded "database," not an in-memory value that resets on reload.
To keep this testable, you MUST include exactly these elements, wired to
real working behavior (style and arrange them however fits the app, but
these ids and behaviors are exact):
- #{REGISTER_USERNAME_ID}, #{REGISTER_PASSWORD_ID} (inputs) and a control
  with id="{REGISTER_SUBMIT_ID}" that creates a new account; reject a
  duplicate username with a visible message instead of crashing or
  silently overwriting the existing account.
- #{LOGIN_USERNAME_ID}, #{LOGIN_PASSWORD_ID} (inputs) and a control with
  id="{LOGIN_SUBMIT_ID}" that logs in an existing account; a wrong
  username or password must NOT log the user in.
- an element with id="{AUTH_STATUS_ID}" whose text content includes the
  logged-in username after a successful login, and never includes any
  registered username before login or after a failed attempt.
"""

QA_PROMPT_ADDENDUM = (
    "This app has accounts: does registration reject a duplicate "
    f"username? Does login reject a wrong password? Does #{AUTH_STATUS_ID} "
    "correctly reflect logged-in/logged-out state? Does localStorage "
    "actually persist accounts rather than losing them on reload?"
)
