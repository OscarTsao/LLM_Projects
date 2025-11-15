from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Protocol, runtime_checkable

try:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    Dataset = DatasetDict = IterableDataset = IterableDatasetDict = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][\w\.\-]*/[A-Za-z0-9][\w\.\-]*$")


class DatasetConfigurationError(RuntimeError):
    """Raised when the dataset configuration is invalid or the dataset cannot be loaded."""


@dataclass
class DatasetConfig:
    id: str
    revision: Optional[str] = None
    splits: Mapping[str, str] = field(
        default_factory=lambda: {"train": "train", "validation": "validation", "test": "test"}
    )
    streaming: bool = False
    cache_dir: Optional[str] = None


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
        if load_dataset is None:
            raise DatasetConfigurationError(
                "The 'datasets' package is not installed. Install it via 'poetry add datasets' "
                "and retry."
            )

        self._validate_config(config)
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
        kwargs: Dict[str, object] = {
            "path": config.id,
            "streaming": config.streaming,
        }
        if config.revision:
            kwargs["revision"] = config.revision
        if config.cache_dir:
            kwargs["cache_dir"] = config.cache_dir
        return kwargs

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
        metadata = {
            "dataset.id": config.id,
            "dataset.revision": config.revision or "default",
            "dataset.splits": ",".join(sorted(available_splits)),
        }
        if resolved_hash:
            metadata["dataset.resolved_fingerprint"] = resolved_hash

        self.logger.info(
            "Loaded dataset %s (revision=%s, splits=%s, fingerprint=%s)",
            config.id,
            config.revision or "default",
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
        revision = config.revision or "default"
        return (
            f"{headline}\n"
            f"Attempted dataset: id='{config.id}', revision='{revision}'.\n"
            f"Details: {details}\n"
            "Remediation: verify the dataset id/revision in 'configs/data/dataset.yaml' or consult "
            "https://huggingface.co/datasets for available configurations."
        )
