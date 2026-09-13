"""Wires the agents together and runs the SDLC pipeline, including the
QA + Testing -> Developer feedback loop (bounded by max_qa_iterations,
default 2 — one Developer pass, then a second pass if QA or Testing found
something to fix, then whatever's produced ships either way). The Testing
stage only does anything when the app has accounts (ArchitectureDoc.has_auth):
for HTML output it drives a real headless browser through register/login
with a synthetic user (see browser_tester.py); for anything else it falls
back to an LLM reasoning through the code (no safe way to execute arbitrary
generated code for other languages here). A longer description or a slower
free model still makes each extra round-trip a real chance to hit a rate
limit, timeout, or truncation — max_qa_iterations stays a small, explicit
cap rather than an unbounded retry loop, even now that it also covers
Testing.
"""

from __future__ import annotations

from typing import Callable

from pipeline.agents import (
    ArchitectAgent,
    DeveloperAgent,
    ProjectManagerAgent,
    QAReviewerAgent,
    RequirementsAnalystAgent,
    TesterAgent,
)
from pipeline.browser_tester import run_browser_auth_test
from pipeline.llm import LLMProvider
from pipeline.schema import ArchitectureDoc, PipelineResult, QAReport, Requirements, TestReport

StageCallback = Callable[[str, str], None]  # (stage_name, status) -> None
# status is one of: "running", "done", "failed"


class Orchestrator:
    def __init__(self, llm: LLMProvider, max_qa_iterations: int = 2):
        self._max_qa_iterations = max_qa_iterations
        self.prompt_log: list[dict] = []
        self._pm = ProjectManagerAgent(llm, self.prompt_log)
        self._requirements_agent = RequirementsAnalystAgent(llm, self.prompt_log)
        self._architect = ArchitectAgent(llm, self.prompt_log)
        self._developer = DeveloperAgent(llm, self.prompt_log)
        self._qa = QAReviewerAgent(llm, self.prompt_log)
        self._tester = TesterAgent(llm, self.prompt_log)

    def _run_tests(self, requirements: Requirements, architecture: ArchitectureDoc, code: str) -> TestReport:
        if not architecture.has_auth:
            return TestReport(passed=True, executed=False, notes=["No accounts in this app; testing stage skipped."])
        if architecture.language == "html":
            return run_browser_auth_test(code)
        return self._tester.test(requirements, architecture, code)

    def run(self, transcript: str, on_stage: StageCallback | None = None) -> PipelineResult:
        def notify(stage: str, status: str) -> None:
            if on_stage:
                on_stage(stage, status)

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
