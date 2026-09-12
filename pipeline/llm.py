"""LLM backends behind one interface, so an agent never knows or cares which
model actually answers it.

OpenRouterProvider is the default (one shared, deployer-supplied API key —
see app.py). MockLLMProvider is deterministic and needs no network/key at
all — it backs the test suite and `cli.py run --mock`.
"""

from __future__ import annotations

import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

from pipeline.secrets import Secrets, mask_key

# Deployer's choice — see docs/PROCESS.md for the full back-and-forth on
# picking one. Free-tier model on OpenRouter; swap this one constant to
# change it, since every agent shares it and there's no per-agent override.
DEFAULT_MODEL = "inclusionai/ling-3.0-flash-vl:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Without an explicit cap, some free-tier models fall back to a small
# provider-side default (sometimes well under what a full self-contained
# HTML/CSS/JS prototype needs) and silently truncate mid-response — no
# error, just an incomplete Developer-stage output. This doesn't force a
# model to use all of it; it just stops our own code from being the
# limiting factor.
MAX_OUTPUT_TOKENS = 16000


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
        headers = {
            "Authorization": f"Bearer {self._secrets.reveal_openrouter_key()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=self._timeout)
        except requests.RequestException as exc:
            raise LLMError(f"Request to OpenRouter failed: {exc}") from None

        if resp.status_code == 401:
            # Printed to the server log (stderr), never to the browser — the
            # masked key/length is enough to spot a stale, truncated, or
            # wrong-key configuration mistake without exposing the key.
            print(
                f"[llm] OpenRouter 401 with OPENROUTER_API_KEY={mask_key(self._secrets.reveal_openrouter_key())}",
                file=sys.stderr,
            )
            raise LLMError(f"OpenRouter rejected the API key (401 Unauthorized): {resp.text[:200]}")
        if resp.status_code == 429:
            raise LLMError("OpenRouter rate-limited this request (429) — try again shortly")
        if resp.status_code >= 400:
            raise LLMError(f"OpenRouter returned {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            # OpenRouter can return HTTP 200 with an error object embedded in
            # the body — e.g. proxying an upstream provider outage — so this
            # has to be checked regardless of resp.status_code, or it falls
            # through to the generic "unexpected shape" branch below with no
            # indication anything upstream actually failed.
            error = data["error"]
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise LLMError(f"OpenRouter upstream error: {message}")
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenRouter response shape: {data!r}") from exc

        if choice.get("finish_reason") == "length":
            # The model hit MAX_OUTPUT_TOKENS (or its own smaller cap) before
            # finishing — `content` here is a real but truncated reply, not
            # an error, so nothing above would have caught it. Silently
            # accepting it is exactly the "stops mid process" bug: the
            # Developer stage would hand QA/the preview a cut-off HTML file
            # with no indication anything went wrong.
            raise LLMError(
                f"OpenRouter cut the reply short (hit the token limit) with model {self._model!r}. "
                "This model's free-tier output limit is too small for this stage. Try again, or "
                "switch to a model with a larger context/output window."
            )
        return content


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
                "screens": [{"name": "Task List", "purpose": "View, add, edit and filter tasks"}],
            }
        if "STAGE: ARCHITECT" in prompt:
            return {
                "tech_approach": "Single self-contained HTML file, vanilla JS, localStorage persistence.",
                "data_model_notes": "One array of task objects keyed by a generated id, stored under one localStorage key.",
                "screen_breakdown": ["Task List (form + filterable table)"],
                "style_notes": "Clean, minimal, readable at 400px width.",
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
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Task Tracker</title>
<style>body{{font-family:sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem}}
.empty-state{{color:#666}}</style></head>
<body>
<h1>Task Tracker</h1>
<form id="taskForm">
  <input name="title" placeholder="title" required>
  <input name="description" placeholder="description">
  <input name="due_date" type="date">
  <select name="priority"><option>Low</option><option>Medium</option><option>High</option></select>
  <button type="submit">add</button>
</form>
<select id="filterPriority"><option value="">All priorities</option><option>Low</option><option>Medium</option><option>High</option></select>
<ul id="taskList"></ul>
{empty_state_markup}
<script>
const KEY = "tasks";
const load = () => JSON.parse(localStorage.getItem(KEY) || "[]");
const save = (tasks) => localStorage.setItem(KEY, JSON.stringify(tasks));
function render() {{
  const filter = document.getElementById("filterPriority").value;
  const tasks = load().filter(t => !filter || t.priority === filter);
  const list = document.getElementById("taskList");
  list.textContent = "";
  const empty = document.querySelector(".empty-state");
  if (empty) empty.hidden = tasks.length !== 0;
  for (const t of tasks) {{
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${{t.title}} (${{t.priority}})${{t.completed ? " [complete]" : ""}}`;
    const completeBtn = document.createElement("button");
    completeBtn.textContent = "complete";
    completeBtn.onclick = () => {{ t.completed = !t.completed; save(load().map(x => x.id === t.id ? t : x)); render(); }};
    const deleteBtn = document.createElement("button");
    deleteBtn.textContent = "delete";
    deleteBtn.onclick = () => {{ save(load().filter(x => x.id !== t.id)); render(); }};
    li.append(label, completeBtn, deleteBtn);
    list.append(li);
  }}
}}
document.getElementById("taskForm").addEventListener("submit", (e) => {{
  e.preventDefault();
  const form = new FormData(e.target);
  const tasks = load();
  tasks.push({{
    id: crypto.randomUUID(),
    title: form.get("title"),
    description: form.get("description"),
    due_date: form.get("due_date"),
    priority: form.get("priority"),
    completed: false,
  }});
  save(tasks);
  e.target.reset();
  render();
}});
document.getElementById("filterPriority").addEventListener("change", render);
render();
</script>
</body></html>"""
