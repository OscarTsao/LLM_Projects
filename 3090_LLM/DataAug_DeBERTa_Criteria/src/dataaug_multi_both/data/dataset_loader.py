from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .dataset import Dataset, DatasetDict, load_hf_dataset as load_dataset

LOGGER = logging.getLogger(__name__)


class DatasetConfigurationError(RuntimeError):
    """Raised when the dataset configuration is invalid or the dataset cannot be loaded."""


@dataclass
class DatasetConfig:
    id: str
    revision: str | None = None
    splits: Mapping[str, str] = field(
        default_factory=lambda: {"train": "train", "validation": "validation", "test": "test"}
    )
    streaming: bool = False
    cache_dir: str | None = None


@runtime_checkable
class _MlflowTagLogger(Protocol):
    def set_tag(self, key: str, value: str) -> None:
        ...


class DatasetLoader:
    """Load and validate study datasets from local CSV files."""

    def __init__(
        self,
        mlflow_client: _MlflowTagLogger | None = None,
        logger: logging.Logger | None = None,
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

        try:
            # Load dataset using local CSV loader
            dataset, metadata = load_dataset(
                dataset_id=config.id,
                revision=config.revision,
                cache_dir=config.cache_dir,
                required_splits=("train", "validation", "test"),
                validate_disjoint=True,
            )
        except ValueError as exc:
            raise DatasetConfigurationError(
                self._build_actionable_error(
                    "Invalid dataset identifier.",
                    config=config,
                    details=str(exc),
                )
            ) from exc
        except FileNotFoundError as exc:
            raise DatasetConfigurationError(
                self._build_actionable_error(
                    "Dataset CSV files could not be found in Data/redsm5.",
                    config=config,
                    details=str(exc),
                )
            ) from exc
        except Exception as exc:  # pragma: no cover - unexpected upstream failure
            raise DatasetConfigurationError(
                self._build_actionable_error(
                    "Unexpected error while loading the dataset.",
                    config=config,
                    details=str(exc),
                )
            ) from exc

        canonical_splits = self._extract_required_splits(dataset, config)
        self._assert_split_disjointness(canonical_splits, config.streaming)
        resolved_hash = metadata.get("resolved_hash")
        self._log_dataset_metadata(config, resolved_hash, set(canonical_splits))
        return canonical_splits

    def _validate_config(self, config: DatasetConfig) -> None:
        if not config.id or not isinstance(config.id, str):
            raise DatasetConfigurationError("Dataset id must be a non-empty string.")
        required = {"train", "validation", "test"}
        missing = required - set(config.splits.keys())
        if missing:
            raise DatasetConfigurationError(
                f"The dataset splits configuration is missing required keys: {sorted(missing)}."
            )

    def _extract_required_splits(self, dataset, config: DatasetConfig):
        dataset_dict = self._ensure_mapping(dataset, config)
        available = set(dataset_dict.keys())
        required = {
            name: alias
            for name, alias in config.splits.items()
            if name in {"train", "validation", "test"}
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
        if isinstance(dataset, DatasetDict):
            return dataset
        if isinstance(dataset, Dataset):
            alias = config.splits["train"]
            return DatasetDict({alias: dataset})
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
            split_name
            for split_name, ds in dataset_dict.items()
            if "post_id" not in ds.column_names
        ]
        if post_id_missing:
            raise DatasetConfigurationError(
                "Each dataset split must include a 'post_id' column for disjointness validation. "
                f"Missing in splits: {sorted(post_id_missing)}."
            )

        seen: set[str] = set()
        duplicates: dict[str, Iterable[str]] = {}
        for split_name, ds in dataset_dict.items():
            post_ids = set(ds["post_id"])
            overlap = seen.intersection(post_ids)
            if overlap:
                duplicates[split_name] = overlap
            seen.update(post_ids)

        if duplicates:
            formatted = ", ".join(
                f"{split}: {sorted(ids)[:5]}{'...' if len(ids) > 5 else ''}"
                for split, ids in duplicates.items()
            )
            raise DatasetConfigurationError(
                f"Dataset splits are not disjoint by 'post_id'. Overlaps detected: {formatted}."
            )

    def _resolve_dataset_hash(self, dataset_dict) -> str | None:
        # For local datasets, create a hash based on dataset size
        import hashlib
        sample_split = dataset_dict.get("train") or next(iter(dataset_dict.values()), None)
        if sample_split is None:
            return None

        # Create hash from dataset sizes
        sizes = {split: len(ds) for split, ds in dataset_dict.items()}
        hash_str = str(sorted(sizes.items()))
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def _log_dataset_metadata(
        self,
        config: DatasetConfig,
        resolved_hash: str | None,
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
        return (
            f"{headline}\n"
            f"Attempted dataset: id='{config.id}'.\n"
            f"Details: {details}\n"
            "Remediation: ensure the dataset CSV files exist in Data/redsm5/ directory."
        )
