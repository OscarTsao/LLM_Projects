"""Flexible data ingestion helpers for real-world datasets."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..candidate_generation import criterion_id
from ..config_schemas import AppConfig
from ..io_utils import read_jsonl
from ..schemas import CriterionSpec, Sample


LOGGER = logging.getLogger(__name__)


def _resolve_dataset_path(cfg: AppConfig, split: str, ext_default: str) -> Path:
    base = Path(cfg.data.path)
    pattern = cfg.data.file_pattern
    if pattern:
        candidate = Path(pattern.format(split=split))
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate
    path_or_name = cfg.data.path_or_name
    if path_or_name:
        candidate = Path(path_or_name)
        if candidate.is_dir():
            candidate = candidate / f"{split}.{ext_default}"
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate
    return base / f"{split}.{ext_default}"


def _group_samples(rows: Iterable[dict[str, object]], cfg: AppConfig, split: str) -> list[Sample]:
    mapping = cfg.data.mapping
    samples: dict[str, dict[str, object]] = {}
    for idx, row in enumerate(rows):
        sample_id = str(row.get(mapping.sample_id) or f"{split}-{idx:06d}")
        note_text = str(row.get(mapping.note) or "")
        if not note_text:
            LOGGER.debug("Row %s missing note text; skipping", idx)
            continue
        criterion_name = str(row.get(mapping.criterion_name) or "")
        if not criterion_name:
            LOGGER.debug("Row %s missing criterion name; skipping", idx)
            continue
        criterion_id_value = str(row.get(mapping.criterion_id) or criterion_id(sample_id, criterion_name))
        definition = str(row.get(mapping.criterion_definition) or criterion_name)
        criterion_meta: dict[str, object] = {}
        for meta_key, column in mapping.metadata.items():
            if column in row:
                criterion_meta[meta_key] = row[column]
        span_start = mapping.evidence_span_start and row.get(mapping.evidence_span_start)
        span_end = mapping.evidence_span_end and row.get(mapping.evidence_span_end)
        gold_spans = None
        if span_start is not None and span_end is not None:
            try:
                start_int = int(span_start)
                end_int = int(span_end)
                if start_int >= 0 and end_int >= start_int:
                    gold_spans = [{"start": start_int, "end": end_int}]
            except (TypeError, ValueError):
                LOGGER.debug("Invalid span for row %s: %s-%s", idx, span_start, span_end)

        sample_entry = samples.setdefault(
            sample_id,
            {
                "id": sample_id,
                "split": split,
                "note_text": note_text,
                "criteria": [],
                "metadata": {"row_count": 0},
            },
        )
        if sample_entry["note_text"] != note_text:
            LOGGER.debug("Conflicting note_text for sample %s; using first occurrence", sample_id)
        sample_entry["metadata"]["row_count"] += 1
        criterion_payload = {
            "id": criterion_id_value,
            "name": criterion_name,
            "definition": definition,
            "metadata": criterion_meta,
        }
        if gold_spans:
            criterion_payload["metadata"]["gold_spans"] = gold_spans
        sample_entry["criteria"].append(criterion_payload)
    return [Sample.model_validate(entry) for entry in samples.values()]


def _load_jsonl(cfg: AppConfig, split: str) -> list[Sample]:
    base = Path(cfg.data.path)
    samples_path = base / f"{split}_samples.jsonl"
    if not samples_path.exists():
        LOGGER.debug("JSONL samples %s missing; returning empty list", samples_path)
        return []
    samples = [Sample.model_validate(row) for row in read_jsonl(samples_path)]
    for sample in samples:
        if not sample.split:
            sample.split = split
    return samples


def _load_csv(cfg: AppConfig, split: str) -> list[Sample]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pandas is required for CSV ingestion") from exc
    source_path = _resolve_dataset_path(cfg, split, "csv")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    df = pd.read_csv(source_path)
    mapping = cfg.data.mapping
    if mapping.split and mapping.split in df.columns:
        df = df[df[mapping.split].astype(str).str.lower() == split.lower()]
    rows = df.to_dict(orient="records")
    return _group_samples(rows, cfg, split)


def _load_parquet(cfg: AppConfig, split: str) -> list[Sample]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for Parquet ingestion") from exc
    source_path = _resolve_dataset_path(cfg, split, "parquet")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    df = pd.read_parquet(source_path)
    mapping = cfg.data.mapping
    if mapping.split and mapping.split in df.columns:
        df = df[df[mapping.split].astype(str).str.lower() == split.lower()]
    rows = df.to_dict(orient="records")
    return _group_samples(rows, cfg, split)


def _load_huggingface(cfg: AppConfig, split: str) -> list[Sample]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required for Hugging Face ingestion") from exc
    if not cfg.data.path_or_name:
        raise RuntimeError("data.path_or_name must be set when using source=huggingface")
    dataset = load_dataset(cfg.data.path_or_name, split=split)
    rows = dataset.to_pandas().to_dict(orient="records")
    return _group_samples(rows, cfg, split)


def load_samples(app_cfg: AppConfig, split: str) -> list[Sample]:
    """Load normalised samples for the requested split."""

    source = app_cfg.data.source.lower()
    if source == "jsonl":
        return _load_jsonl(app_cfg, split)
    if source == "csv":
        return _load_csv(app_cfg, split)
    if source == "parquet":
        return _load_parquet(app_cfg, split)
    if source == "huggingface":
        return _load_huggingface(app_cfg, split)
    raise ValueError(f"Unsupported data.source '{app_cfg.data.source}'")


__all__ = ["load_samples"]
