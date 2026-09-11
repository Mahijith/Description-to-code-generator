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
    screens: list[Screen] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "description": self.description,
            "entities": [e.to_dict() for e in self.entities],
            "actions": self.actions,
            "filters": self.filters,
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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureDoc":
        return cls(
            tech_approach=str(data.get("tech_approach", "")),
            data_model_notes=str(data.get("data_model_notes", "")),
            screen_breakdown=list(data.get("screen_breakdown", []) or []),
            style_notes=str(data.get("style_notes", "")),
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
class PipelineResult:
    brief: ProjectBrief
    requirements: Requirements
    architecture: ArchitectureDoc
    html: str
    qa_reports: list[QAReport]
    iterations: int
    summary: str
    prompt_log: list[dict]

    def to_dict(self) -> dict:
        return {
            "brief": self.brief.to_dict(),
            "requirements": self.requirements.to_dict(),
            "architecture": self.architecture.to_dict(),
            "html": self.html,
            "qa_reports": [q.to_dict() for q in self.qa_reports],
            "iterations": self.iterations,
            "summary": self.summary,
            "prompt_log": self.prompt_log,
        }
