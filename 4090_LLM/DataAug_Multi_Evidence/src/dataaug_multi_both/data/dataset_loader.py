from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

LOGGER = logging.getLogger(__name__)

DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][\w\.\-]*/[A-Za-z0-9][\w\.\-]*$")
LOCAL_DATASET_BUILDERS = {
    "csv",
    "json",
    "jsonl",
    "parquet",
    "text",
}


class Dataset:
    """Custom dataset class to replace HuggingFace Dataset."""

    def __init__(self, data: pd.DataFrame, fingerprint: str | None = None) -> None:
        """
        Initialize dataset with pandas DataFrame.

        Parameters
        ----------
        data:
            Pandas DataFrame containing the dataset
        fingerprint:
            Optional fingerprint for caching/versioning
        """
        self._data = data
        self._fingerprint = fingerprint or str(hash(str(data.shape)))
        self.column_names = list(data.columns)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: int | str | list) -> Any:
        if isinstance(key, str):
            # Column access
            return self._data[key].tolist()
        elif isinstance(key, int):
            # Row access
            return self._data.iloc[key].to_dict()
        elif isinstance(key, list):
            # Multiple row indices
            return [self._data.iloc[i].to_dict() for i in key]
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def select(self, indices: list[int]) -> "Dataset":
        """Select rows by indices."""
        selected_data = self._data.iloc[indices].reset_index(drop=True)
        return Dataset(selected_data, fingerprint=self._fingerprint)

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return self._data.copy()


class DatasetDict(dict):
    """Custom dataset dict to replace HuggingFace DatasetDict."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class IterableDataset(Dataset):
    """Custom iterable dataset (same as Dataset for local files)."""

    pass


class IterableDatasetDict(DatasetDict):
    """Custom iterable dataset dict (same as DatasetDict for local files)."""

    pass


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
    data_files: Any | None = None
    split_percentages: Mapping[str, float] | None = None
    config_dir: Path | None = field(default=None, repr=False, compare=False)


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
            dataset = self._load_local_dataset(config)
        except FileNotFoundError as exc:
            raise DatasetConfigurationError(
                self._build_actionable_error(
                    "Dataset file could not be found.",
                    config=config,
                    details=str(exc),
                )
            ) from exc
        except (RuntimeError, OSError, IOError, pd.errors.ParserError) as exc:
            raise DatasetConfigurationError(
                self._build_actionable_error(
                    "Error while loading the dataset.",
                    config=config,
                    details=str(exc),
                )
            ) from exc

        canonical_splits = self._extract_required_splits(dataset, config)
        self._assert_split_disjointness(canonical_splits, config.streaming)
        resolved_hash = self._resolve_dataset_hash(canonical_splits)
        self._log_dataset_metadata(config, resolved_hash, set(canonical_splits))
        return canonical_splits

    def _load_local_dataset(self, config: DatasetConfig) -> DatasetDict:
        """Load dataset from local CSV file."""
        # Resolve data file path
        data_file_path = self._resolve_data_file_path(config)

        # Load CSV with pandas
        df = pd.read_csv(data_file_path)

        # Create Dataset object
        dataset = Dataset(df)

        # Return as DatasetDict with single split (will be split later if needed)
        alias = config.splits.get("train", "train")
        return DatasetDict({alias: dataset})

    def _validate_config(self, config: DatasetConfig) -> None:
        if not config.id or not isinstance(config.id, str):
            raise DatasetConfigurationError("Dataset id must be a non-empty string.")
        if config.id not in LOCAL_DATASET_BUILDERS:
            raise DatasetConfigurationError(
                f"Dataset id '{config.id}' must be a local dataset builder (e.g. 'csv', 'json')."
            )
        if not config.data_files:
            raise DatasetConfigurationError(
                "data_files must be specified for local dataset loading."
            )
        required = {"train", "validation", "test"}
        missing = required - set(config.splits.keys())
        if missing:
            raise DatasetConfigurationError(
                f"The dataset splits configuration is missing required keys: {sorted(missing)}."
            )

    def _resolve_data_file_path(self, config: DatasetConfig) -> Path:
        """Resolve the data file path from config."""
        if not config.data_files:
            raise DatasetConfigurationError("data_files must be specified")

        # Handle different data_files formats
        data_file = config.data_files
        if isinstance(data_file, dict):
            # Get train split file
            data_file = data_file.get("train", data_file.get(list(data_file.keys())[0]))
        elif isinstance(data_file, list):
            data_file = data_file[0]

        if not isinstance(data_file, str):
            raise DatasetConfigurationError(f"Invalid data_files format: {type(data_file)}")

        # Normalize path
        file_path = Path(data_file)
        if not file_path.is_absolute() and config.config_dir:
            file_path = (config.config_dir / file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        return file_path

    def _extract_required_splits(self, dataset, config: DatasetConfig):
        dataset_dict = self._ensure_mapping(dataset, config)
        available = set(dataset_dict.keys())
        required = {
            name: alias
            for name, alias in config.splits.items()
            if name in {"train", "validation", "test"}
        }

        missing = [name for name, alias in required.items() if alias not in available]

        # If splits are missing and we have split_percentages, create splits from single dataset
        if missing and config.split_percentages:
            self.logger.info("Creating splits from single dataset using split_percentages")
            return self._create_splits_from_single_dataset(dataset_dict, config)

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

    def _create_splits_from_single_dataset(self, dataset_dict, config: DatasetConfig):
        """Create train/validation/test splits from a single dataset, ensuring post_id disjointness."""
        # Get the single dataset (should only have one split, typically 'train')
        if len(dataset_dict) != 1:
            raise DatasetConfigurationError(
                "split_percentages can only be used when loading a single dataset split"
            )

        single_split = next(iter(dataset_dict.values()))

        if config.streaming:
            raise DatasetConfigurationError(
                "Cannot create splits from streaming datasets. Set streaming=false."
            )

        # Get split percentages
        if config.split_percentages is None:  # Narrow Optional for type checker
            raise DatasetConfigurationError(
                "split_percentages must be provided when creating splits from a single dataset"
            )
        sp = config.split_percentages
        train_pct = sp.get("train", 0.7)
        val_pct = sp.get("validation", 0.15)
        test_pct = sp.get("test", 0.15)

        # Validate percentages sum to 1.0
        total = train_pct + val_pct + test_pct
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise DatasetConfigurationError(f"Split percentages must sum to 1.0, got {total}")

        # Get unique post_ids and shuffle them with isolated random state
        post_ids = list(set(single_split["post_id"]))
        rng = random.Random(42)  # Use isolated instance for reproducibility
        rng.shuffle(post_ids)

        # Calculate split boundaries based on number of posts
        n_posts = len(post_ids)
        train_size = int(n_posts * train_pct)
        val_size = int(n_posts * val_pct)

        # Assign posts to splits
        train_post_ids = set(post_ids[:train_size])
        val_post_ids = set(post_ids[train_size : train_size + val_size])
        test_post_ids = set(post_ids[train_size + val_size :])

        # Filter dataset by post_id
        train_indices = [
            i for i, pid in enumerate(single_split["post_id"]) if pid in train_post_ids
        ]
        val_indices = [i for i, pid in enumerate(single_split["post_id"]) if pid in val_post_ids]
        test_indices = [i for i, pid in enumerate(single_split["post_id"]) if pid in test_post_ids]

        train_ds = single_split.select(train_indices)
        val_ds = single_split.select(val_indices)
        test_ds = single_split.select(test_indices)

        self.logger.info(
            f"Created splits by post_id: train={len(train_ds)} ({len(train_post_ids)} posts), "
            f"validation={len(val_ds)} ({len(val_post_ids)} posts), "
            f"test={len(test_ds)} ({len(test_post_ids)} posts)"
        )

        return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

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
        duplicates: dict[str, set[str]] = {}
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
        # Prefer fingerprints, fall back to dataset info hash
        sample_split = dataset_dict.get("train") or next(iter(dataset_dict.values()), None)
        if sample_split is None:
            return None

        fingerprint: str | None = getattr(sample_split, "_fingerprint", None)
        if fingerprint:
            return str(fingerprint)

        info = getattr(sample_split, "info", None)
        if info:
            hash_value: str | None = getattr(info, "hash", None)
            if hash_value:
                return str(hash_value)
        return None

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
            "Remediation: verify the dataset configuration in 'configs/data/dataset.yaml' "
            "and ensure data_files points to a valid local CSV file."
        )


def build_dataset_config_from_dict(
    section: Mapping[str, Any] | None,
    *,
    config_dir: Path | None = None,
) -> DatasetConfig:
    """Create a :class:`DatasetConfig` from a raw mapping (e.g. YAML data)."""

    section = dict(section or {})

    raw_splits = section.get("splits") or {}
    if not raw_splits:
        raw_splits = {"train": "train", "validation": "validation", "test": "test"}
    else:
        raw_splits = {str(k): str(v) for k, v in dict(raw_splits).items()}

    raw_percentages = section.get("split_percentages")
    if isinstance(raw_percentages, Mapping):
        split_percentages = {str(k): float(v) for k, v in raw_percentages.items()}
    else:
        split_percentages = None

    data_files = section.get("data_files")

    return DatasetConfig(
        id=str(section.get("id", "csv")),
        revision=section.get("revision"),
        splits=raw_splits,
        streaming=bool(section.get("streaming", False)),
        cache_dir=section.get("cache_dir"),
        data_files=data_files,
        split_percentages=split_percentages,
        config_dir=config_dir,
    )
