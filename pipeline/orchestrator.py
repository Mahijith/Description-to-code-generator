"""Wires the agents together and runs the SDLC pipeline, including the
QA -> Developer feedback loop.
"""

from __future__ import annotations

from typing import Callable

from pipeline.agents import ArchitectAgent, DeveloperAgent, ProjectManagerAgent, QAReviewerAgent, RequirementsAnalystAgent
from pipeline.llm import LLMProvider
from pipeline.schema import PipelineResult, QAReport

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
        html = ""
        qa_feedback: list[str] | None = None
        iteration = 0
        while True:
            iteration += 1
            notify(f"developer (pass {iteration})", "running")
            html = self._developer.build(requirements, architecture, qa_feedback)
            notify(f"developer (pass {iteration})", "done")

            notify(f"qa (pass {iteration})", "running")
            qa_report = self._qa.review(requirements, architecture, html)
            notify(f"qa (pass {iteration})", "done" if qa_report.passed else "failed")
            qa_reports.append(qa_report)

            if not self._pm.decide(qa_report, iteration, self._max_qa_iterations):
                break
            qa_feedback = qa_report.issues

        notify("pm_summary", "running")
        summary = self._pm.summarize(brief, requirements, qa_reports, iteration)
        notify("pm_summary", "done")

        return PipelineResult(
            brief=brief,
            requirements=requirements,
            architecture=architecture,
            html=html,
            qa_reports=qa_reports,
            iterations=iteration,
            summary=summary,
            prompt_log=self.prompt_log,
        )
