"""LLM backends behind one interface, so an agent never knows or cares which
model actually answers it.

AIHubMixProvider is the default in app.py (one shared, deployer-supplied
API key — see app.py) — added after OpenRouter's shared key kept hitting
its rate limit under real testing. OpenRouterProvider remains fully
implemented and tested; both talk to an OpenAI-compatible chat-completions
endpoint, so they share one request/error-handling implementation
(`_complete_via_openai_compatible_api`) and differ only in URL, key
source, and default model. MockLLMProvider is deterministic and needs no
network/key at all — it backs the test suite and `cli.py run --mock`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

from pipeline.secrets import Secrets, mask_key

# Free-tier model on OpenRouter; swap this one constant to change it, since
# every agent shares it and there's no per-agent override. See
# docs/PROCESS.md for the back-and-forth on picking one.
DEFAULT_MODEL = "inclusionai/ling-3.0-flash-vl:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# AIHubMix model id inferred from the model's catalog page URL
# (aihubmix.com/model/ling-3.0-flash-free) — aihubmix.com is unreachable
# from this sandbox to verify the exact API string directly, so if a real
# run rejects this model id, that's the first thing to check.
DEFAULT_AIHUBMIX_MODEL = "ling-3.0-flash-free"
AIHUBMIX_URL = "https://aihubmix.com/v1/chat/completions"


class LLMError(RuntimeError):
    """Raised on any provider failure. Never carries a raw API key."""


class LLMProvider(ABC):
    """Ask a model for text, or for one JSON value."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's raw text reply to `prompt`."""

    def test_connection(self) -> str:
        """One cheap call to confirm the key/model combo actually works,
        before spending a full 5-agent run on it. Returns the raw reply on
        success; raises LLMError (with the real cause) on failure.
        """
        return self.complete("Reply with exactly one word: OK")

    def complete_json(self, prompt: str) -> dict | list:
        """Return the reply parsed as JSON, tolerant of prose around it."""
        raw = self.complete(prompt)
        parsed = _extract_json(raw)
        if parsed is None:
            # One retry with a sharper instruction before giving up.
            retry_prompt = (
                prompt
                + "\n\nYour previous reply could not be parsed as JSON. "
                "Reply with ONLY the JSON value — no prose, no markdown fences."
            )
            raw = self.complete(retry_prompt)
            parsed = _extract_json(raw)
        if parsed is None:
            raise LLMError(f"Model reply was not valid JSON: {raw[:300]!r}")
        return parsed


def _extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    start_candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if start_candidates:
        start = min(start_candidates)
        end_candidates = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
        if end_candidates:
            end = max(end_candidates) + 1
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _complete_via_openai_compatible_api(
    *, url: str, api_key: str, model: str, prompt: str, timeout: int, provider_name: str, key_env_var: str,
) -> str:
    """Shared request/error-handling for any OpenAI-chat-completions-shaped
    provider (OpenRouter, AIHubMix, ...): same payload shape, same failure
    modes worth surfacing distinctly — a rejected key, a rate limit, an
    error object embedded in an HTTP 200 body (a provider proxying an
    upstream outage), and a reply cut off before finishing. This app never
    sends its own token limit, so a "length" finish_reason always means the
    model or provider hit *its own* maximum, not a restriction this app
    imposed — see docs/PROCESS.md for the rounds of debugging that shaped
    every one of these checks.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError(f"Request to {provider_name} failed: {exc}") from None

    if resp.status_code == 401:
        # Printed to the server log (stderr), never to the browser — the
        # masked key/length is enough to spot a stale, truncated, or
        # wrong-key configuration mistake without exposing the key.
        print(f"[llm] {provider_name} 401 with {key_env_var}={mask_key(api_key)}", file=sys.stderr)
        raise LLMError(f"{provider_name} rejected the API key (401 Unauthorized): {resp.text[:200]}")
    if resp.status_code == 429:
        raise LLMError(f"{provider_name} rate-limited this request (429) — try again shortly")
    if resp.status_code >= 400:
        raise LLMError(f"{provider_name} returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        error = data["error"]
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise LLMError(f"{provider_name} upstream error: {message}")
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected {provider_name} response shape: {data!r}") from exc

    if choice.get("finish_reason") == "length":
        usage = data.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        detail = f"the model produced {completion_tokens} tokens before stopping" if completion_tokens is not None else "the reply was cut off"
        raise LLMError(
            f"{provider_name} cut the reply short with model {model!r}: {detail}. This app sends no "
            "token limit of its own, so the cutoff came from the model/provider's own maximum. Try "
            "again, or switch to a different model."
        )
    return content


class OpenRouterProvider(LLMProvider):
    """Talks to OpenRouter's OpenAI-compatible chat-completions endpoint.

    A single OpenRouter key covers every model routed through it, so this
    one class serves every agent in the pipeline — there is no per-agent
    key or per-agent provider.
    """

    def __init__(self, secrets: Secrets, model: str = DEFAULT_MODEL, timeout: int = 120):
        if not secrets.has_openrouter_key:
            raise LLMError("No OpenRouter API key configured")
        self._secrets = secrets
        self._model = model
        self._timeout = timeout

    def complete(self, prompt: str) -> str:
        return _complete_via_openai_compatible_api(
            url=OPENROUTER_URL,
            api_key=self._secrets.reveal_openrouter_key(),
            model=self._model,
            prompt=prompt,
            timeout=self._timeout,
            provider_name="OpenRouter",
            key_env_var="OPENROUTER_API_KEY",
        )


class AIHubMixProvider(LLMProvider):
    """Talks to AIHubMix's OpenAI-compatible chat-completions endpoint —
    an alternate free-tier gateway, added after OpenRouter's shared key
    kept hitting its rate limit under real testing.
    """

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_AIHUBMIX_MODEL, timeout: int = 120):
        self._api_key = (api_key or os.environ.get("AIHUBMIX_API_KEY") or "").strip()
        if not self._api_key:
            raise LLMError("No AIHubMix API key configured")
        self._model = model
        self._timeout = timeout

    def complete(self, prompt: str) -> str:
        return _complete_via_openai_compatible_api(
            url=AIHUBMIX_URL,
            api_key=self._api_key,
            model=self._model,
            prompt=prompt,
            timeout=self._timeout,
            provider_name="AIHubMix",
            key_env_var="AIHUBMIX_API_KEY",
        )


@dataclass
class _MockScript:
    """A canned answer keyed by a substring that must appear in the prompt."""

    marker: str
    response: object  # str for complete(), dict/list for complete_json()


class MockLLMProvider(LLMProvider):
    """Deterministic stand-in: no network, no key, no randomness.

    Sniffs the prompt for a marker each agent includes (its stage name) and
    returns a fixed answer. The QA stage is scripted to fail once so the
    Developer -> QA feedback loop is genuinely exercised by "Demo mode" and
    by the test suite, without needing a real model to actually miss
    anything.
    """

    def __init__(self) -> None:
        self._qa_calls = 0

    def complete(self, prompt: str) -> str:
        if "STAGE: DEVELOPER" in prompt:
            return self._developer_html(prompt)
        if "STAGE: PM_SUMMARY" in prompt:
            return (
                "Project complete. The requirements analyst identified the core entity "
                "and required actions; the architect chose a single-file HTML/JS/"
                "localStorage prototype; QA found one gap on the first pass "
                "(missing empty-state messaging) which the developer addressed on the "
                "second pass. Shipped after 2 iteration(s)."
            )
        raise LLMError(f"MockLLMProvider has no scripted text answer for prompt: {prompt[:120]!r}")

    def complete_json(self, prompt: str):
        if "STAGE: PM_KICKOFF" in prompt:
            return {
                "goal": "Prototype the application described in the transcript.",
                "scope": "A single primary entity with create/edit/delete/complete/filter.",
                "out_of_scope": "Authentication, multi-user sync, backend persistence.",
                "success_criteria": [
                    "Every entity field from the transcript is editable in the UI",
                    "Every action mentioned (add/edit/delete/complete/filter) works",
                ],
            }
        if "STAGE: REQUIREMENTS" in prompt:
            return {
                "app_name": "Task Tracker",
                "description": "Track personal tasks with due dates and priority.",
                "entities": [
                    {
                        "name": "Task",
                        "fields": [
                            {"name": "title", "type": "text"},
                            {"name": "description", "type": "text"},
                            {"name": "due_date", "type": "date"},
                            {"name": "priority", "type": "select", "options": ["Low", "Medium", "High"]},
                            {"name": "completed", "type": "boolean"},
                        ],
                    }
                ],
                "actions": ["add", "edit", "delete", "complete", "filter"],
                "filters": ["priority", "completed"],
                "features": [],
                "screens": [{"name": "Task List", "purpose": "View, add, edit and filter tasks"}],
            }
        if "STAGE: ARCHITECT" in prompt:
            return {
                "tech_approach": "Single self-contained HTML file, vanilla JS, localStorage persistence.",
                "data_model_notes": "One array of task objects keyed by a generated id, stored under one localStorage key.",
                "screen_breakdown": ["Task List (form + filterable table)"],
                "style_notes": "Clean, minimal, readable at 400px width.",
                "has_auth": False,
            }
        if "STAGE: QA" in prompt:
            self._qa_calls += 1
            if self._qa_calls == 1:
                return {"passed": False, "issues": ["No empty-state message is shown when the task list is empty."]}
            return {"passed": True, "issues": []}
        raise LLMError(f"MockLLMProvider has no scripted JSON answer for prompt: {prompt[:120]!r}")

    def _developer_html(self, prompt: str) -> str:
        needs_empty_state = "empty-state" in prompt.lower() or "empty state" in prompt.lower()
        empty_state_markup = (
            '<p class="empty-state" hidden>No tasks yet — add one above.</p>' if needs_empty_state else ""
        )
        # Field/filter ids below match pipeline.crud_contract's slugify()
        # exactly (e.g. "due_date" -> "due-date") — this fixture exists so
        # the orchestrator's real browser-driven CRUD test has something
        # deterministic and correct to exercise end to end.
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Task Tracker</title>
<style>body{{font-family:sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem}}
.empty-state{{color:#666}}</style></head>
<body>
<h1>Task Tracker</h1>
<form id="add-form">
  <input id="field-title" placeholder="title" required>
  <input id="field-description" placeholder="description">
  <input id="field-due-date" type="date">
  <select id="field-priority"><option>Low</option><option>Medium</option><option>High</option></select>
  <button type="submit" id="add-submit">Add</button>
</form>
<select id="filter-priority"><option value="">All priorities</option><option>Low</option><option>Medium</option><option>High</option></select>
<ul id="item-list"></ul>
{empty_state_markup}
<script>
const KEY = "tasks";
let editingId = null;
const load = () => JSON.parse(localStorage.getItem(KEY) || "[]");
const save = (tasks) => localStorage.setItem(KEY, JSON.stringify(tasks));
function render() {{
  const filter = document.getElementById("filter-priority").value;
  const tasks = load().filter(t => !filter || t.priority === filter);
  const list = document.getElementById("item-list");
  list.textContent = "";
  const empty = document.querySelector(".empty-state");
  if (empty) empty.hidden = tasks.length !== 0;
  for (const t of tasks) {{
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${{t.title}} - ${{t.description}} (${{t.due_date}}, ${{t.priority}})${{t.completed ? " [complete]" : ""}}`;
    const completeBtn = document.createElement("button");
    completeBtn.textContent = "complete";
    completeBtn.onclick = () => {{ t.completed = !t.completed; save(load().map(x => x.id === t.id ? t : x)); render(); }};
    const editBtn = document.createElement("button");
    editBtn.textContent = "edit";
    editBtn.setAttribute("data-action", "edit");
    editBtn.onclick = () => {{
      editingId = t.id;
      document.getElementById("field-title").value = t.title;
      document.getElementById("field-description").value = t.description;
      document.getElementById("field-due-date").value = t.due_date;
      document.getElementById("field-priority").value = t.priority;
      document.getElementById("add-submit").textContent = "Save";
    }};
    const deleteBtn = document.createElement("button");
    deleteBtn.textContent = "delete";
    deleteBtn.setAttribute("data-action", "delete");
    deleteBtn.onclick = () => {{ save(load().filter(x => x.id !== t.id)); render(); }};
    li.append(label, completeBtn, editBtn, deleteBtn);
    list.append(li);
  }}
}}
document.getElementById("add-form").addEventListener("submit", (e) => {{
  e.preventDefault();
  const values = {{
    title: document.getElementById("field-title").value,
    description: document.getElementById("field-description").value,
    due_date: document.getElementById("field-due-date").value,
    priority: document.getElementById("field-priority").value,
  }};
  let tasks = load();
  if (editingId) {{
    tasks = tasks.map(t => t.id === editingId ? {{...t, ...values}} : t);
    editingId = null;
    document.getElementById("add-submit").textContent = "Add";
  }} else {{
    tasks.push({{ id: crypto.randomUUID(), completed: false, ...values }});
  }}
  save(tasks);
  e.target.reset();
  render();
}});
document.getElementById("filter-priority").addEventListener("change", render);
render();
</script>
</body></html>"""
