"""Schema definitions shared across candidate generation, judging, and training."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class SchemaEncoder(BaseModel):
    """Base schema capable of serialising itself to JSON/Dict."""

    model_config = {"extra": "ignore"}

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def to_json(self) -> str:
        return self.model_dump_json()


class CriterionSpec(SchemaEncoder):
    """Structured description of a clinical criterion."""

    id: str
    name: str
    definition: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def display_text(self) -> str:
        if self.definition:
            return self.definition
        return self.name


class EvidenceSpan(SchemaEncoder):
    """Character-level span highlighting supporting evidence."""

    candidate_index: int
    start: int
    end: int
    confidence: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Sample(SchemaEncoder):
    """Normalised sample used throughout the pipeline."""

    id: str
    split: str = "train"
    note_text: str
    criteria: list[CriterionSpec]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_criteria(cls, values: dict[str, Any]) -> dict[str, Any]:
        criteria = values.get("criteria", [])
        if criteria and isinstance(criteria[0], str):
            note_id = values.get("id", "sample")
            converted = []
            for idx, name in enumerate(criteria):
                cid = f"{note_id}-{idx:04d}"
                converted.append({"id": cid, "name": name, "definition": name})
            values["criteria"] = converted
        values.setdefault("split", values.get("metadata", {}).get("split", "train"))
        return values


class Candidate(SchemaEncoder):
    """Candidate snippet extracted from a sample."""

    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    score: Optional[float] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Preference(SchemaEncoder):
    """Pairwise preference between two candidates."""

    winner_idx: int
    loser_idx: int
    weight: float = 1.0


class JudgingJob(SchemaEncoder):
    """Payload sent to the judge provider."""

    job_id: str
    note_id: str
    criterion_id: str
    criterion: CriterionSpec
    criterion_text: str
    note_text: str
    candidates: list[Candidate]
    seed: Optional[int] = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _ensure_criterion(cls, values: dict[str, Any]) -> dict[str, Any]:
        criterion_obj = values.get("criterion")
        if isinstance(criterion_obj, dict):
            pass
        elif criterion_obj is None:
            criterion_text = values.get("criterion_text", "")
            criterion_id = values.get("criterion_id", "")
            values["criterion"] = {
                "id": criterion_id or f"{values.get('job_id', 'job')}-criterion",
                "name": criterion_text or criterion_id or "criterion",
                "definition": criterion_text or criterion_id or "criterion",
            }
        else:
            criterion_text = getattr(criterion_obj, "display_text", None) or getattr(criterion_obj, "definition", None)
            if criterion_text:
                values.setdefault("criterion_text", criterion_text)
        if "criterion_text" not in values:
            criterion = values.get("criterion")
            if isinstance(criterion, dict):
                values["criterion_text"] = criterion.get("definition") or criterion.get("name")
        return values
        return values


class Judgment(SchemaEncoder):
    """Structured response from a judge provider."""

    job_id: str
    note_id: str
    criterion_id: str
    criterion: CriterionSpec
    criterion_text: str
    note_text: str
    candidates: list[Candidate]
    best_idx: int
    preferences: list[Preference]
    ranking: list[int] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    rationale: Optional[str] = None
    provider: str
    model: Optional[str] = None
    latency_s: Optional[float] = None
    token_usage: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _ensure_criterion(cls, values: dict[str, Any]) -> dict[str, Any]:
        criterion_obj = values.get("criterion")
        if isinstance(criterion_obj, dict):
            pass
        elif criterion_obj is None:
            criterion_text = values.get("criterion_text", "")
            criterion_id = values.get("criterion_id", "")
            values["criterion"] = {
                "id": criterion_id or f"{values.get('job_id', 'job')}-criterion",
                "name": criterion_text or criterion_id or "criterion",
                "definition": criterion_text or criterion_id or "criterion",
            }
        else:
            criterion_text = getattr(criterion_obj, "display_text", None) or getattr(criterion_obj, "definition", None)
            if criterion_text:
                values.setdefault("criterion_text", criterion_text)
        if "criterion_text" not in values:
            criterion = values.get("criterion")
            if isinstance(criterion, dict):
                values["criterion_text"] = criterion.get("definition") or criterion.get("name")
        values.setdefault("ranking", values.get("ranking", []))
        values.setdefault("preferences", values.get("preferences", []))
        values.setdefault("evidence_spans", values.get("evidence_spans", []))
        values.setdefault("ranking", values.get("ranking", []))
        values.setdefault("preferences", values.get("preferences", []))
        values.setdefault("evidence_spans", values.get("evidence_spans", []))
        return values


class DPORecord(SchemaEncoder):
    """Preference record suitable for TRL DPO / RRHF training."""

    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "Candidate",
    "CriterionSpec",
    "DPORecord",
    "EvidenceSpan",
    "JudgingJob",
    "Judgment",
    "Preference",
    "Sample",
    "SchemaEncoder",
]
