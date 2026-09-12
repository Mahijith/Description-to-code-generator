"""The five SDLC personas. Each agent is a thin, stateless wrapper around an
LLMProvider: it knows how to phrase its own job as a prompt and how to shape
the reply, and nothing else. All the repetitive parts (building the prompt
scaffold, calling the model, recording what was asked) live once in Agent.
"""

from __future__ import annotations

import json
from abc import ABC

from pipeline.llm import LLMProvider
from pipeline.schema import ArchitectureDoc, ProjectBrief, QAReport, Requirements


class Agent(ABC):
    """Shared plumbing for every agent: no system prompt exists on most
    backends, so every call spells out persona + task + data + output shape
    in one message. Every prompt sent is recorded (stage, prompt) in the
    shared log the Orchestrator hands to each agent — that log is exactly
    the "prompts I experimented with" evidence for the process write-up.
    """

    def __init__(self, llm: LLMProvider, prompt_log: list[dict]):
        self._llm = llm
        self._log = prompt_log

    def _ask(self, stage: str, prompt: str) -> str:
        self._log.append({"stage": stage, "prompt": prompt})
        return self._llm.complete(prompt)

    def _ask_json(self, stage: str, prompt: str):
        self._log.append({"stage": stage, "prompt": prompt})
        return self._llm.complete_json(prompt)


class ProjectManagerAgent(Agent):
    """Kicks the project off, and decides when the QA/Developer loop ends."""

    def kickoff(self, transcript: str) -> ProjectBrief:
        prompt = f"""You are the Project Manager on a small software team.
This is STAGE: PM_KICKOFF.

A stakeholder described an application out loud; here is the transcript:
\"\"\"{transcript}\"\"\"

Write a short project brief. Reply with ONLY a JSON object:
{{"goal": string, "scope": string, "out_of_scope": string, "success_criteria": [string, ...]}}"""
        data = self._ask_json("pm_kickoff", prompt)
        return ProjectBrief.from_dict(data)

    def decide(self, qa_report: QAReport, iteration: int, max_iterations: int) -> bool:
        """True = send it back to the Developer for another pass."""
        return (not qa_report.passed) and iteration < max_iterations

    def summarize(
        self,
        brief: ProjectBrief,
        requirements: Requirements,
        qa_reports: list[QAReport],
        iterations: int,
    ) -> str:
        prompt = f"""You are the Project Manager on a small software team.
This is STAGE: PM_SUMMARY.

Project brief: {json.dumps(brief.to_dict())}
Requirements delivered: {json.dumps(requirements.to_dict())}
QA history (one report per iteration): {json.dumps([q.to_dict() for q in qa_reports])}
Total iterations taken: {iterations}

Write a short (3-5 sentence) sign-off summary of what was built, what QA
found along the way, and the final state. Plain text, no markdown."""
        return self._ask("pm_summary", prompt).strip()


class RequirementsAnalystAgent(Agent):
    def extract(self, transcript: str, brief: ProjectBrief) -> Requirements:
        prompt = f"""You are the Requirements Analyst on a small software team.
This is STAGE: REQUIREMENTS.

Project brief: {json.dumps(brief.to_dict())}
Full transcript:
\"\"\"{transcript}\"\"\"

Extract the functional requirements for a rapid prototype. Identify the
single primary entity being managed (e.g. "Task", "Contact", "Recipe"),
its fields, the actions the user described (e.g. add, edit, delete,
complete, filter, search, sort), any filters, and the screens needed.

Reply with ONLY a JSON object of this exact shape:
{{
  "app_name": string,
  "description": string,
  "entities": [{{"name": string, "fields": [{{"name": string, "type": "text"|"number"|"date"|"boolean"|"select", "options": [string, ...]}}]}}],
  "actions": [string, ...],
  "filters": [string, ...],
  "screens": [{{"name": string, "purpose": string}}]
}}
Only the primary entity is required; "options" may be omitted or empty for
non-select fields."""
        data = self._ask_json("requirements", prompt)
        return Requirements.from_dict(data)


class ArchitectAgent(Agent):
    def design(self, requirements: Requirements) -> ArchitectureDoc:
        prompt = f"""You are the Software Architect on a small software team.
This is STAGE: ARCHITECT.

Requirements: {json.dumps(requirements.to_dict())}

Design the technical approach for a RAPID PROTOTYPE (not production
software). Choose the SINGLE BEST language for THIS specific app — you are
not restricted to any one language or platform. For example: a form-driven
CRUD app that a person would open and click around in is usually best as
one self-contained HTML file (inline CSS/JS, `localStorage` persistence,
opens directly in a browser, no server); a data-processing, automation, or
command-line-style tool described as something you'd "run" rather than
"open" is usually better as one self-contained Python script (standard
library only). Pick whichever matches how the app is actually meant to be
used — do not default to HTML out of habit. Whatever you choose, the whole
prototype must still be ONE self-contained file: no build step, no
third-party dependencies to install, no external network calls. Decide the
data model (how the primary entity is stored) and a short screen/output
breakdown.

Reply with ONLY a JSON object:
{{"tech_approach": string, "data_model_notes": string, "screen_breakdown": [string, ...], "style_notes": string, "language": string, "file_extension": string}}
"language" must be a Pygments-recognized language id matching your choice
(e.g. "html", "python", "javascript"). "file_extension" must match it with
no leading dot (e.g. "html", "py", "js")."""
        data = self._ask_json("architect", prompt)
        return ArchitectureDoc.from_dict(data)


class DeveloperAgent(Agent):
    def build(self, requirements: Requirements, architecture: ArchitectureDoc, qa_feedback: list[str] | None = None) -> str:
        feedback_block = ""
        if qa_feedback:
            feedback_block = (
                "\nThe previous draft was reviewed by QA and needs these fixes:\n- "
                + "\n- ".join(qa_feedback)
            )
        prompt = f"""You are the Developer on a small software team.
This is STAGE: DEVELOPER.

Requirements: {json.dumps(requirements.to_dict())}
Architecture: {json.dumps(architecture.to_dict())}
{feedback_block}

Write the COMPLETE prototype as ONE self-contained {architecture.language}
file matching the architecture above exactly: no build step, no
third-party dependencies to install, no external network calls. Implement
every field of the primary entity and every action listed in requirements
(e.g. add/edit/delete/mark complete/filter/search), and show a friendly
empty-state message or output when there's nothing to show yet. If the
chosen language is HTML: inline CSS and JS, `localStorage` persistence,
and render dynamic content with `textContent`, never by concatenating
user input into `innerHTML`. For any language: never build a shell
command, SQL query, or markup string by concatenating untrusted input —
use safe APIs (parameterized queries, escaped/templated output, etc.)
instead. No explanation text, no markdown fences.

Reply with ONLY the raw source code for that one file."""
        code = self._ask("developer", prompt)
        return _strip_code_fence(code)


class QAReviewerAgent(Agent):
    def review(self, requirements: Requirements, architecture: ArchitectureDoc, code: str) -> QAReport:
        prompt = f"""You are the QA Reviewer on a small software team.
This is STAGE: QA.

Requirements: {json.dumps(requirements.to_dict())}
Architecture: {json.dumps(architecture.to_dict())}
Generated prototype ({architecture.language} source):
\"\"\"{code}\"\"\"

Check the prototype against the requirements: is every field of the
primary entity present as an input/parameter? Does every action (add/
edit/delete/complete/filter/etc.) actually work in the code? Is there a
message or output shown when there's nothing to show yet? Is user input
handled safely for this language (no string-concatenated shell command,
SQL query, or markup — e.g. no `innerHTML` built from untrusted input)?

Reply with ONLY a JSON object:
{{"passed": boolean, "issues": [string, ...]}}
"passed" is true only if there are no issues."""
        data = self._ask_json("qa", prompt)
        return QAReport.from_dict(data)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()
