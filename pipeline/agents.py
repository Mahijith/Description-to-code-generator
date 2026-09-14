"""The five SDLC personas. Each agent is a thin, stateless wrapper around an
LLMProvider: it knows how to phrase its own job as a prompt and how to shape
the reply, and nothing else. All the repetitive parts (building the prompt
scaffold, calling the model, recording what was asked) live once in Agent.
"""

from __future__ import annotations

import json
from abc import ABC

from pipeline import auth_contract, crud_contract
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

    def decide(self, qa_report: QAReport, test_report: TestReport, iteration: int, max_iterations: int) -> bool:
        """True = send it back to the Developer for another pass."""
        return (not qa_report.passed or not test_report.passed) and iteration < max_iterations

    def summarize(
        self,
        brief: ProjectBrief,
        requirements: Requirements,
        qa_reports: list[QAReport],
        test_reports: list[TestReport],
        iterations: int,
    ) -> str:
        prompt = f"""You are the Project Manager on a small software team.
This is STAGE: PM_SUMMARY.

Project brief: {json.dumps(brief.to_dict())}
Requirements delivered: {json.dumps(requirements.to_dict())}
QA history (one report per iteration): {json.dumps([q.to_dict() for q in qa_reports])}
Testing history (one report per iteration): {json.dumps([t.to_dict() for t in test_reports])}
Total iterations taken: {iterations}

Write a short (3-5 sentence) sign-off summary of what was built, what QA
and Testing found along the way, and the final state. Plain text, no
markdown."""
        return self._ask("pm_summary", prompt).strip()


class RequirementsAnalystAgent(Agent):
    def extract(self, transcript: str, brief: ProjectBrief) -> Requirements:
        prompt = f"""You are the Requirements Analyst on a small software team.
This is STAGE: REQUIREMENTS.

Project brief: {json.dumps(brief.to_dict())}
Full transcript:
\"\"\"{transcript}\"\"\"

Extract the functional requirements for a rapid prototype — whatever
shape this specific app actually needs. Don't force it into a
data-management mold it doesn't fit:

- If the app manages a collection of records with clear fields (e.g.
  tasks, contacts, recipes, inventory items), describe that as a primary
  entity (name + fields) and put the record-level actions (add, edit,
  delete, complete, filter, search, sort) in "actions".
- If the app doesn't manage a record collection at all (e.g. a tool, a
  game, a calculator, a visualizer, a debugging aid, a chat interface),
  leave "entities" and "actions" empty and instead describe every real
  capability the app needs in "features" — plain-language statements of
  what it must actually do (e.g. "let the user paste a code snippet and
  step through it line by line," "highlight the currently executing
  line," "show a call stack that updates as execution advances").
- Many apps are a genuine mix of both — describe whatever combination is
  actually true. Never invent a "primary entity" or generic CRUD verbs
  just to fill in a field; leave it empty when it doesn't apply.

Reply with ONLY a JSON object of this exact shape:
{{
  "app_name": string,
  "description": string,
  "entities": [{{"name": string, "fields": [{{"name": string, "type": "text"|"number"|"date"|"boolean"|"select", "options": [string, ...]}}]}}],
  "actions": [string, ...],
  "filters": [string, ...],
  "features": [string, ...],
  "screens": [{{"name": string, "purpose": string}}]
}}
Nothing is required except "app_name" and "description" — leave any of
"entities"/"actions"/"filters"/"features" empty if it genuinely doesn't
apply; "options" may be omitted or empty for non-select fields."""
        data = self._ask_json("requirements", prompt)
        return Requirements.from_dict(data)


class ArchitectAgent(Agent):
    def design(self, requirements: Requirements) -> ArchitectureDoc:
        prompt = f"""You are the Software Architect on a small software team.
This is STAGE: ARCHITECT.

Requirements: {json.dumps(requirements.to_dict())}

Design the technical approach for a RAPID PROTOTYPE (not production
software). It will be built as ONE self-contained HTML file: inline CSS
and JS, `localStorage` persistence, opens directly in a browser, no
server, no build step, no third-party dependencies, no external network
calls. Decide how the app's data or state is held, if it manages any, and
a short screen/output breakdown.

Also decide whether this app's concept genuinely implies user accounts —
each person seeing only their own saved data, an explicit sign-up/log-in,
or a multi-user tool. Most rapid prototypes do NOT need this (a calculator,
a single shared list, a converter) — only set it true when accounts
actually fit the concept described.

Reply with ONLY a JSON object:
{{"tech_approach": string, "data_model_notes": string, "screen_breakdown": [string, ...], "style_notes": string, "has_auth": boolean}}"""
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
        auth_block = auth_contract.DEVELOPER_PROMPT_BLOCK if architecture.has_auth else ""
        crud_block = crud_contract.developer_prompt_block(requirements)
        prompt = f"""You are the Developer on a small software team.
This is STAGE: DEVELOPER.

Requirements: {json.dumps(requirements.to_dict())}
Architecture: {json.dumps(architecture.to_dict())}
{feedback_block}
{auth_block}
{crud_block}

Write the COMPLETE prototype as ONE self-contained HTML file matching the
architecture above exactly: inline CSS and JS, `localStorage`
persistence where the app genuinely needs to remember something, no
build step, no third-party dependencies to install, no external network
calls. Implement everything Requirements actually describes, to the best
of your ability: every field/action for the primary entity if there is
one, AND every capability listed in "features". Build whatever UI and
interaction pattern actually fits THIS app — do not bolt on a generic
add/edit/delete/filter data-entry form for an app that isn't about
managing a list of records. If the app does show a list of records, show
a friendly empty-state message when there's nothing to show yet. Render
dynamic content with `textContent`, never by concatenating user input
into `innerHTML`. No explanation text, no markdown fences.

Reply with ONLY the raw source code for that one file."""
        code = self._ask("developer", prompt)
        return _strip_code_fence(code)


class QAReviewerAgent(Agent):
    def review(self, requirements: Requirements, architecture: ArchitectureDoc, code: str) -> QAReport:
        has_entity = requirements.primary_entity is not None
        entity_line = (
            "Is every field of the primary entity present as an input, and does "
            "every listed action (add/edit/delete/complete/filter/etc.) actually "
            "work? Is there a message shown when there's nothing to show yet? "
            if has_entity
            else ""
        )
        features_line = (
            "Does the prototype actually implement every capability listed in "
            "'features', correctly and completely for what this app is? "
            if requirements.features
            else ""
        )
        auth_line = f"\n{auth_contract.QA_PROMPT_ADDENDUM}" if architecture.has_auth else ""
        crud_line = f"\n{crud_contract.qa_prompt_addendum(requirements)}" if has_entity else ""
        prompt = f"""You are the Code Reviewer on a small software team.
This is STAGE: QA.

Requirements: {json.dumps(requirements.to_dict())}
Architecture: {json.dumps(architecture.to_dict())}
Generated prototype (HTML source):
\"\"\"{code}\"\"\"

Check the prototype against the requirements: {entity_line}{features_line}Is
user input handled safely (no `innerHTML` built from untrusted input —
use `textContent` instead)?{crud_line}{auth_line}

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
