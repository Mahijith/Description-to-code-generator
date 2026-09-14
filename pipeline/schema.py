"""Value objects passed between pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ProjectBrief:
    goal: str
    scope: str
    out_of_scope: str
    success_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectBrief":
        return cls(
            goal=str(data.get("goal", "")),
            scope=str(data.get("scope", "")),
            out_of_scope=str(data.get("out_of_scope", "")),
            success_criteria=list(data.get("success_criteria", []) or []),
        )


@dataclass
class Field:
    name: str
    type: str  # "text" | "number" | "date" | "boolean" | "select"
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Field":
        return cls(
            name=str(data.get("name", "value")),
            type=str(data.get("type", "text")),
            options=list(data.get("options", []) or []),
        )


@dataclass
class Entity:
    name: str
    fields: list[Field] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "fields": [f.to_dict() for f in self.fields]}

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(
            name=str(data.get("name", "Item")),
            fields=[Field.from_dict(f) for f in (data.get("fields") or [])],
        )


@dataclass
class Screen:
    name: str
    purpose: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Screen":
        return cls(name=str(data.get("name", "")), purpose=str(data.get("purpose", "")))


@dataclass
class Requirements:
    app_name: str
    description: str
    entities: list[Entity] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)  # freeform capabilities for apps that aren't a records list
    screens: list[Screen] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "description": self.description,
            "entities": [e.to_dict() for e in self.entities],
            "actions": self.actions,
            "filters": self.filters,
            "features": self.features,
            "screens": [s.to_dict() for s in self.screens],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Requirements":
        return cls(
            app_name=str(data.get("app_name", "My App")),
            description=str(data.get("description", "")),
            entities=[Entity.from_dict(e) for e in (data.get("entities") or [])],
            actions=list(data.get("actions", []) or []),
            filters=list(data.get("filters", []) or []),
            features=list(data.get("features", []) or []),
            screens=[Screen.from_dict(s) for s in (data.get("screens") or [])],
        )

    @property
    def primary_entity(self) -> Entity | None:
        return self.entities[0] if self.entities else None


@dataclass
class ArchitectureDoc:
    tech_approach: str
    data_model_notes: str
    screen_breakdown: list[str] = field(default_factory=list)
    style_notes: str = ""
    language: str = "html"  # a Pygments/st.code language id, e.g. "html", "python"
    file_extension: str = "html"  # no leading dot, e.g. "html", "py"
    has_auth: bool = False  # true when this app's concept implies user accounts

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureDoc":
        return cls(
            tech_approach=str(data.get("tech_approach", "")),
            data_model_notes=str(data.get("data_model_notes", "")),
            screen_breakdown=list(data.get("screen_breakdown", []) or []),
            style_notes=str(data.get("style_notes", "")),
            language=str(data.get("language") or "html"),
            file_extension=str(data.get("file_extension") or "html"),
            has_auth=bool(data.get("has_auth", False)),
        )


@dataclass
class QAReport:
    passed: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QAReport":
        return cls(passed=bool(data.get("passed", False)), issues=list(data.get("issues", []) or []))


@dataclass
class TestReport:
    passed: bool
    notes: list[str] = field(default_factory=list)
    executed: bool = False  # True when a real (e.g. browser) execution produced this, not just an LLM's say-so

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TestReport":
        return cls(passed=bool(data.get("passed", False)), notes=list(data.get("notes", []) or []))


@dataclass
class PipelineResult:
    brief: ProjectBrief
    requirements: Requirements
    architecture: ArchitectureDoc
    code: str
    qa_reports: list[QAReport]
    test_reports: list[TestReport]
    iterations: int
    summary: str
    prompt_log: list[dict]

    def to_dict(self) -> dict:
        return {
            "brief": self.brief.to_dict(),
            "requirements": self.requirements.to_dict(),
            "architecture": self.architecture.to_dict(),
            "code": self.code,
            "qa_reports": [q.to_dict() for q in self.qa_reports],
            "test_reports": [t.to_dict() for t in self.test_reports],
            "iterations": self.iterations,
            "summary": self.summary,
            "prompt_log": self.prompt_log,
        }
