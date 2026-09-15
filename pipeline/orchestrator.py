"""Wires the agents together and runs the SDLC pipeline, including the
Code Review + Testing -> Developer feedback loop (bounded by
max_qa_iterations, default 3 — the Developer gets up to two retries if
Code Review or Testing found something to fix, then whatever's produced
ships either way). Every app is one self-contained HTML file (see
ArchitectAgent), so Testing always drives a real headless browser through
the app's core add/edit/delete/filter flow (pipeline/crud_contract.py),
plus registration/login when the app has accounts
(ArchitectureDoc.has_auth, pipeline/auth_contract.py) — see
browser_tester.py. Real execution costs no LLM requests at all, so the
iteration cap is sized around the LLM calls only: a longer description or
a slower free model still makes each extra round-trip a real chance to
hit a rate limit, timeout, or truncation, so it stays a small, explicit
cap rather than an unbounded retry loop.

A dedicated Scope Gate runs before anything else, including PM kickoff
(see ScopeGateAgent) — if it finds no application/process/feature to
build at all (e.g. the transcript is just a stray greeting, an off-topic
remark, or a bare "build me an app" with nothing named), `run` returns
immediately: PM kickoff, Requirements, Architect, Developer, Code
Review, and Testing never run at all, and no PipelineResult.architecture/
code is produced.
"""

from __future__ import annotations

from typing import Callable

from pipeline.agents import (
    ArchitectAgent,
    DeveloperAgent,
    ProjectManagerAgent,
    QAReviewerAgent,
    RequirementsAnalystAgent,
    ScopeGateAgent,
)
from pipeline.browser_tester import run_browser_functional_test
from pipeline.llm import LLMProvider
from pipeline.schema import ArchitectureDoc, PipelineResult, QAReport, Requirements, TestReport

StageCallback = Callable[[str, str], None]  # (stage_name, status) -> None
# status is one of: "running", "done", "failed"


class Orchestrator:
    def __init__(self, llm: LLMProvider, max_qa_iterations: int = 3):
        self._max_qa_iterations = max_qa_iterations
        self.prompt_log: list[dict] = []
        self._scope_gate = ScopeGateAgent(llm, self.prompt_log)
        self._pm = ProjectManagerAgent(llm, self.prompt_log)
        self._requirements_agent = RequirementsAnalystAgent(llm, self.prompt_log)
        self._architect = ArchitectAgent(llm, self.prompt_log)
        self._developer = DeveloperAgent(llm, self.prompt_log)
        self._qa = QAReviewerAgent(llm, self.prompt_log)

    def _run_tests(self, requirements: Requirements, architecture: ArchitectureDoc, code: str) -> TestReport:
        return run_browser_functional_test(requirements, architecture.has_auth, code)

    def run(self, transcript: str, on_stage: StageCallback | None = None) -> PipelineResult:
        def notify(stage: str, status: str) -> None:
            if on_stage:
                on_stage(stage, status)

        notify("scope_gate", "running")
        scope_check = self._scope_gate.check(transcript)
        notify("scope_gate", "done" if scope_check.has_scope else "failed")

        if not scope_check.has_scope:
            # Nothing to build (a greeting, off-topic content, a bare
            # instruction with nothing named) — stop before even PM
            # kickoff runs. No extra LLM call needed for the summary;
            # the Scope Gate's own reason already explains it.
            return PipelineResult(
                summary=scope_check.reason or (
                    "No application, process, or feature to build was found in "
                    "this description — there's nothing here to turn into a "
                    "prototype."
                ),
            )

        notify("pm_kickoff", "running")
        brief = self._pm.kickoff(transcript)
        notify("pm_kickoff", "done")

        notify("requirements", "running")
        requirements = self._requirements_agent.extract(transcript, brief)
        notify("requirements", "done")

        notify("architect", "running")
        architecture = self._architect.design(requirements)
        notify("architect", "done")

        qa_reports: list[QAReport] = []
        test_reports: list[TestReport] = []
        code = ""
        qa_feedback: list[str] | None = None
        iteration = 0
        while True:
            iteration += 1
            notify(f"developer (pass {iteration})", "running")
            code = self._developer.build(requirements, architecture, qa_feedback)
            notify(f"developer (pass {iteration})", "done")

            notify(f"qa (pass {iteration})", "running")
            qa_report = self._qa.review(requirements, architecture, code)
            notify(f"qa (pass {iteration})", "done" if qa_report.passed else "failed")
            qa_reports.append(qa_report)

            notify(f"testing (pass {iteration})", "running")
            test_report = self._run_tests(requirements, architecture, code)
            notify(f"testing (pass {iteration})", "done" if test_report.passed else "failed")
            test_reports.append(test_report)

            if not self._pm.decide(qa_report, test_report, iteration, self._max_qa_iterations):
                break
            qa_feedback = list(qa_report.issues)
            if not test_report.passed:
                qa_feedback += [f"Testing found: {note}" for note in test_report.notes]

        notify("pm_summary", "running")
        summary = self._pm.summarize(brief, requirements, qa_reports, test_reports, iteration)
        notify("pm_summary", "done")

        return PipelineResult(
            brief=brief,
            requirements=requirements,
            architecture=architecture,
            code=code,
            qa_reports=qa_reports,
            test_reports=test_reports,
            iterations=iteration,
            summary=summary,
            prompt_log=self.prompt_log,
        )
