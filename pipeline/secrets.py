"""Encapsulates the OpenRouter API key so it can't leak by accident.

The raw key is read in exactly one place in the whole codebase: inside
OpenRouterProvider, when it builds the Authorization header for a request.
Everywhere else — logs, prompt history, exception messages, Streamlit
widgets — only ever sees the masked form.
"""

from __future__ import annotations


class Secrets:
    """Holds one API key. Never prints, logs, or serializes the raw value."""

    def __init__(self, openrouter_api_key: str | None):
        self._openrouter_api_key = (openrouter_api_key or "").strip()

    @property
    def has_openrouter_key(self) -> bool:
        return bool(self._openrouter_api_key)

    def reveal_openrouter_key(self) -> str:
        """The only method that returns the raw key. Callers must not log it."""
        if not self._openrouter_api_key:
            raise ValueError("No OpenRouter API key configured")
        return self._openrouter_api_key

    def _masked(self) -> str:
        key = self._openrouter_api_key
        if not key:
            return "<no key>"
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"

    def __repr__(self) -> str:  # never leak the raw key via repr/logging
        return f"Secrets(openrouter_api_key={self._masked()!r})"

    __str__ = __repr__
