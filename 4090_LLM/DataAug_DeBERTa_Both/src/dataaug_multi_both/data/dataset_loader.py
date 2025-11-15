from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable

import pandas as pd

try:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    Dataset = DatasetDict = IterableDataset = IterableDatasetDict = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][\w\.\-]*/[A-Za-z0-9][\w\.\-]*$")
DEFAULT_CRITERIA_ORDER: Sequence[str] = (
    "ANHEDONIA",
    "APPETITE_CHANGE",
    "COGNITIVE_ISSUES",
    "DEPRESSED_MOOD",
    "FATIGUE",
    "PSYCHOMOTOR",
    "SLEEP_ISSUES",
    "SPECIAL_CASE",
    "SUICIDAL_THOUGHTS",
    "WORTHLESSNESS",
)


class DatasetConfigurationError(RuntimeError):
    """Raised when the dataset configuration is invalid or the dataset cannot be loaded."""


@dataclass
class DatasetConfig:
    id: Optional[str] = None
    revision: Optional[str] = None
    splits: Mapping[str, str] = field(
        default_factory=lambda: {"train": "train", "validation": "validation", "test": "test"}
    )
    streaming: bool = False
    cache_dir: Optional[str] = None
    local_data: Optional[Mapping[str, object]] = None


@runtime_checkable
class _MlflowTagLogger(Protocol):
    def set_tag(self, key: str, value: str) -> None:
        ...


class DatasetLoader:
    """Load and validate study datasets from the Hugging Face Hub."""

    def __init__(
        self,
        mlflow_client: Optional[_MlflowTagLogger] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Parameters
        ----------
        mlflow_client:
            Minimal MLflow tagging interface. It must expose a
            ``set_tag(key: str, value: str) -> None`` method. The default implementation
            uses :mod:`mlflow` if it is available.
        logger:
            Optional custom logger. Defaults to the module logger.
        """

        self.logger = logger or LOGGER
        self.mlflow_client = mlflow_client

    def load(self, config: DatasetConfig):
        """Load the dataset according to ``config`` and perform validation checks."""
        self._validate_config(config)
        if config.local_data:
            dataset = self._load_local_dataset(config)
        else:
            if load_dataset is None:
                raise DatasetConfigurationError(
                    "The 'datasets' package is not installed. Install it via 'poetry add datasets' "
                    "and retry."
                )

            dataset_kwargs = self._build_dataset_kwargs(config)

            try:
                dataset = load_dataset(**dataset_kwargs)
            except ValueError as exc:
                raise DatasetConfigurationError(
                    self._build_actionable_error(
                        "Invalid dataset identifier or unavailable configuration.",
                        config=config,
                        details=str(exc),
                    )
                ) from exc
            except FileNotFoundError as exc:
                raise DatasetConfigurationError(
                    self._build_actionable_error(
                        "Dataset revision or files could not be found on the Hugging Face Hub.",
                        config=config,
                        details=str(exc),
                    )
                ) from exc
            except Exception as exc:  # pragma: no cover - unexpected upstream failure
                raise DatasetConfigurationError(
                    self._build_actionable_error(
                        "Unexpected error while downloading the dataset.",
                        config=config,
                        details=str(exc),
                    )
                ) from exc

        canonical_splits = self._extract_required_splits(dataset, config)
        self._assert_split_disjointness(canonical_splits, config.streaming)
        resolved_hash = self._resolve_dataset_hash(canonical_splits)
        self._log_dataset_metadata(config, resolved_hash, set(canonical_splits))
        return canonical_splits

    def _validate_config(self, config: DatasetConfig) -> None:
        if config.local_data:
            if not isinstance(config.local_data, Mapping):
                raise DatasetConfigurationError("local_data must be a mapping of configuration values.")
            required_keys = {"posts_file", "annotations_file", "groundtruth_file"}
            missing_local = required_keys - set(config.local_data.keys())
            if missing_local:
                raise DatasetConfigurationError(
                    f"Local dataset configuration is missing required keys: {sorted(missing_local)}."
                )
        else:
            if not config.id or not isinstance(config.id, str):
                raise DatasetConfigurationError("Dataset id must be a non-empty string.")
            if not DATASET_ID_PATTERN.match(config.id):
                raise DatasetConfigurationError(
                    f"Dataset id '{config.id}' is not valid. Expected format like 'owner/dataset-name'."
                )
        required = {"train", "validation", "test"}
        missing = required - set(config.splits.keys())
        if missing:
            raise DatasetConfigurationError(
                f"The dataset splits configuration is missing required keys: {sorted(missing)}."
            )

    def _build_dataset_kwargs(self, config: DatasetConfig) -> Dict[str, object]:
        if not config.id:
            raise DatasetConfigurationError(
                "Dataset id must be provided for remote datasets. Set local_data for local sources."
            )
        kwargs: Dict[str, object] = {
            "path": config.id,
            "streaming": config.streaming,
        }
        if config.revision:
            kwargs["revision"] = config.revision
        if config.cache_dir:
            kwargs["cache_dir"] = config.cache_dir
        return kwargs

    def _load_local_dataset(self, config: DatasetConfig):
        if Dataset is None or DatasetDict is None:
            raise DatasetConfigurationError(
                "The 'datasets' package is not installed. Install it via 'poetry add datasets' "
                "and retry."
            )

        assert config.local_data is not None  # for mypy
        posts_file = Path(str(config.local_data.get("posts_file", ""))).expanduser()
        annotations_file = Path(str(config.local_data.get("annotations_file", ""))).expanduser()
        groundtruth_file = Path(str(config.local_data.get("groundtruth_file", ""))).expanduser()

        for path, label in [
            (posts_file, "posts_file"),
            (annotations_file, "annotations_file"),
            (groundtruth_file, "groundtruth_file"),
        ]:
            if not path.exists():
                raise DatasetConfigurationError(
                    f"Local dataset {label} not found at '{path}'. Verify the path in configuration."
                )

        try:
            posts_df = pd.read_csv(posts_file)
        except Exception as exc:  # pragma: no cover - depends on local files
            raise DatasetConfigurationError(
                f"Failed to read posts CSV at '{posts_file}': {exc}"
            ) from exc

        try:
            groundtruth_df = pd.read_csv(groundtruth_file)
        except Exception as exc:  # pragma: no cover - depends on local files
            raise DatasetConfigurationError(
                f"Failed to read ground truth CSV at '{groundtruth_file}': {exc}"
            ) from exc

        try:
            annotations_df = pd.read_csv(annotations_file)
        except Exception as exc:  # pragma: no cover - depends on local files
            raise DatasetConfigurationError(
                f"Failed to read annotations CSV at '{annotations_file}': {exc}"
            ) from exc

        if "post_id" not in posts_df.columns or "text" not in posts_df.columns:
            raise DatasetConfigurationError(
                f"Posts file '{posts_file}' must contain 'post_id' and 'text' columns."
            )
        if "post_id" not in groundtruth_df.columns:
            raise DatasetConfigurationError(
                f"Ground truth file '{groundtruth_file}' must contain 'post_id'."
            )

        missing_criteria = [col for col in DEFAULT_CRITERIA_ORDER if col not in groundtruth_df.columns]
        if missing_criteria:
            raise DatasetConfigurationError(
                f"Ground truth file '{groundtruth_file}' is missing criteria columns: {missing_criteria}."
            )

        posts_text_map = {row["post_id"]: row["text"] for _, row in posts_df.iterrows()}
        annotations_grouped = annotations_df.groupby("post_id") if not annotations_df.empty else {}

        split_ratios_cfg = config.local_data.get("split_ratios", {})  # type: ignore[assignment]
        if not isinstance(split_ratios_cfg, Mapping):
            raise DatasetConfigurationError("local_data.split_ratios must be a mapping of split -> ratio.")
        if not split_ratios_cfg:
            split_ratios_cfg = {"train": 0.7, "validation": 0.15, "test": 0.15}
        required_splits = {"train", "validation", "test"}
        missing_split_ratios = required_splits - set(split_ratios_cfg.keys())
        if missing_split_ratios:
            raise DatasetConfigurationError(
                f"Split ratios are missing required keys: {sorted(missing_split_ratios)}."
            )
        total_ratio = sum(float(split_ratios_cfg[name]) for name in required_splits)
        if total_ratio <= 0:
            raise DatasetConfigurationError("Sum of split ratios must be positive.")
        split_ratios = {name: float(split_ratios_cfg[name]) / total_ratio for name in required_splits}

        split_seed = int(config.local_data.get("split_seed", 1337))

        def assign_split(post_identifier: str) -> str:
            digest = hashlib.sha256(f"{post_identifier}-{split_seed}".encode("utf-8")).digest()
            value = int.from_bytes(digest[:8], "big") / 2**64
            cumulative = 0.0
            for split_name in ("train", "validation", "test"):
                cumulative += split_ratios[split_name]
                if value < cumulative:
                    return split_name
            return "test"

        splits_records: Dict[str, list] = {name: [] for name in required_splits}

        for _, row in groundtruth_df.iterrows():
            post_id = row["post_id"]
            text = posts_text_map.get(post_id, row.get("text", ""))
            if not isinstance(text, str) or not text:
                raise DatasetConfigurationError(
                    f"Post text missing for post_id '{post_id}'. Ensure posts CSV includes matching entries."
                )

            criteria_labels = []
            for col in DEFAULT_CRITERIA_ORDER:
                value = row[col]
                if pd.isna(value):
                    criteria_labels.append(0)
                else:
                    try:
                        criteria_labels.append(int(value))
                    except (TypeError, ValueError):
                        raise DatasetConfigurationError(
                            f"Invalid label value '{value}' for criterion '{col}' in post '{post_id}'."
                        ) from None

            if isinstance(annotations_grouped, dict):
                post_annotations = annotations_grouped.get(post_id, pd.DataFrame())
            else:
                try:
                    post_annotations = annotations_grouped.get_group(post_id)
                except KeyError:
                    post_annotations = pd.DataFrame()

            evidence_records = []
            if not post_annotations.empty:
                for _, ann in post_annotations.iterrows():
                    status_raw = ann.get("status", 0)
                    try:
                        status_int = int(status_raw)
                    except (TypeError, ValueError):
                        status_int = 0
                    evidence_records.append(
                        {
                            "sentence_id": str(ann.get("sentence_id", "")),
                            "sentence_text": ann.get("sentence_text", "") or "",
                            "symptom": ann.get("DSM5_symptom", ""),
                            "status": status_int,
                            "explanation": ann.get("explanation", ""),
                        }
                    )

            split_name = assign_split(post_id)
            record = {
                "post_id": post_id,
                "text": text,
                "criteria_labels": criteria_labels,
                "criteria_label_names": list(DEFAULT_CRITERIA_ORDER),
                "evidence_annotations": evidence_records,
                "positive_evidence_sentences": [
                    evidence["sentence_text"] for evidence in evidence_records if evidence["status"] == 1
                ],
            }
            splits_records[split_name].append(record)

        for split_name, records in splits_records.items():
            if not records:
                raise DatasetConfigurationError(
                    f"Local dataset split '{split_name}' has no records. Adjust split ratios or seed."
                )

        dataset = DatasetDict(
            {split_name: Dataset.from_list(records) for split_name, records in splits_records.items()}
        )
        return dataset

    def _extract_required_splits(self, dataset, config: DatasetConfig):
        dataset_dict = self._ensure_mapping(dataset, config)
        available = set(dataset_dict.keys())
        required = {
            name: alias for name, alias in config.splits.items() if name in {"train", "validation", "test"}
        }

        missing = [name for name, alias in required.items() if alias not in available]
        if missing:
            raise DatasetConfigurationError(
                self._build_actionable_error(
                    "Required dataset splits are missing.",
                    config=config,
                    details=(
                        f"Required canonical splits: {sorted(required.keys())}; "
                        f"configured source splits: {required}; "
                        f"available splits: {sorted(available)}."
                    ),
                )
            )

        canonical = type(dataset_dict)()  # preserve DatasetDict vs IterableDatasetDict
        for canonical_name, source_name in required.items():
            canonical[canonical_name] = dataset_dict[source_name]
        return canonical

    def _ensure_mapping(self, dataset, config: DatasetConfig):
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            return dataset
        if isinstance(dataset, (Dataset, IterableDataset)):
            alias = config.splits["train"]
            if isinstance(dataset, Dataset):
                return DatasetDict({alias: dataset})
            return IterableDatasetDict({alias: dataset})
        raise DatasetConfigurationError(
            self._build_actionable_error(
                "Dataset loader returned an unsupported object type.",
                config=config,
                details=f"Received {type(dataset)!r}",
            )
        )

    def _assert_split_disjointness(self, dataset_dict, streaming: bool) -> None:
        if streaming:
            self.logger.info("Skipping post_id disjointness validation for streaming datasets.")
            return

        post_id_missing = [
            split_name for split_name, ds in dataset_dict.items() if "post_id" not in ds.column_names
        ]
        if post_id_missing:
            raise DatasetConfigurationError(
                "Each dataset split must include a 'post_id' column for disjointness validation. "
                f"Missing in splits: {sorted(post_id_missing)}."
            )

        seen: set[str] = set()
        duplicates: Dict[str, Iterable[str]] = {}
        for split_name, ds in dataset_dict.items():
            post_ids = set(ds["post_id"])
            overlap = seen.intersection(post_ids)
            if overlap:
                duplicates[split_name] = overlap
            seen.update(post_ids)

        if duplicates:
            formatted = ", ".join(
                f"{split}: {sorted(list(ids))[:5]}{'...' if len(ids) > 5 else ''}"
                for split, ids in duplicates.items()
            )
            raise DatasetConfigurationError(
                f"Dataset splits are not disjoint by 'post_id'. Overlaps detected: {formatted}."
            )

    def _resolve_dataset_hash(self, dataset_dict) -> Optional[str]:
        # Prefer fingerprints, fall back to dataset info hash
        sample_split = dataset_dict.get("train") or next(iter(dataset_dict.values()), None)
        if sample_split is None:
            return None

        fingerprint = getattr(sample_split, "_fingerprint", None)
        if fingerprint:
            return fingerprint

        info = getattr(sample_split, "info", None)
        if info and getattr(info, "hash", None):
            return info.hash  # type: ignore[attr-defined]
        return None

    def _log_dataset_metadata(
        self,
        config: DatasetConfig,
        resolved_hash: Optional[str],
        available_splits: Iterable[str],
    ) -> None:
        dataset_id = config.id or "local-dataset"
        dataset_revision = config.revision or ("local" if config.local_data else "default")
        metadata = {
            "dataset.id": dataset_id,
            "dataset.revision": dataset_revision,
            "dataset.splits": ",".join(sorted(available_splits)),
        }
        if resolved_hash:
            metadata["dataset.resolved_fingerprint"] = resolved_hash

        self.logger.info(
            "Loaded dataset %s (revision=%s, splits=%s, fingerprint=%s)",
            dataset_id,
            dataset_revision,
            sorted(available_splits),
            resolved_hash or "n/a",
        )

        # If an explicit MLflow client is provided, use it. Otherwise, attempt to use mlflow globally.
        if self.mlflow_client is not None:
            for key, value in metadata.items():
                self.mlflow_client.set_tag(key, value)  # type: ignore[attr-defined]
            return

        try:
            import mlflow

            if mlflow.active_run() is not None:
                for key, value in metadata.items():
                    mlflow.set_tag(key, value)
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            self.logger.debug("MLflow not installed; skipping dataset metadata tagging.")

    def _build_actionable_error(self, headline: str, config: DatasetConfig, details: str) -> str:
        revision = config.revision or ("local" if config.local_data else "default")
        dataset_id = config.id or "local-dataset"
        return (
            f"{headline}\n"
            f"Attempted dataset: id='{dataset_id}', revision='{revision}'.\n"
            f"Details: {details}\n"
            "Remediation: verify the dataset configuration in 'configs/data/redsm5.yaml' or update "
            "local_data paths if using a local copy."
        )
