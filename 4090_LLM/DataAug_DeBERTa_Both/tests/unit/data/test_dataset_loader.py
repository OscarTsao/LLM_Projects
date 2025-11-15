import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

import src.dataaug_multi_both.data.dataset_loader as dataset_loader
from src.dataaug_multi_both.data.dataset_loader import (
    DatasetConfig,
    DatasetConfigurationError,
    DatasetLoader,
)


class StubSplit:
    def __init__(self, post_ids: List[str], fingerprint: str = "stub-hash") -> None:
        self._post_ids = post_ids
        self._fingerprint = fingerprint
        self.column_names = ["post_id", "text"]
        self.info = SimpleNamespace(hash="info-hash")

    def __getitem__(self, key: str) -> List[str]:
        if key == "post_id":
            return self._post_ids
        raise KeyError(key)

    @classmethod
    def from_list(cls, records: List[Dict[str, object]]) -> "StubSplit":
        post_ids = [str(record["post_id"]) for record in records]
        return cls(post_ids)


class StubDatasetDict(dict):
    """Minimal stand-in for datasets.DatasetDict."""


class StubIterableDatasetDict(dict):
    """Minimal stand-in for datasets.IterableDatasetDict."""


class DummyMlflow:
    def __init__(self) -> None:
        self.tags: Dict[str, str] = {}

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value


@pytest.fixture(autouse=True)
def patch_dataset_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset_loader, "DatasetDict", StubDatasetDict)
    monkeypatch.setattr(dataset_loader, "IterableDatasetDict", StubIterableDatasetDict)
    monkeypatch.setattr(dataset_loader, "Dataset", StubSplit)
    monkeypatch.setattr(dataset_loader, "IterableDataset", StubSplit)


@pytest.mark.unit
def test_load_dataset_success(monkeypatch: pytest.MonkeyPatch) -> None:
    splits = {
        "train": StubSplit(["1", "2"]),
        "validation": StubSplit(["3"]),
        "test": StubSplit(["4"]),
    }
    stub_dict = StubDatasetDict(
        {
            "train": splits["train"],
            "validation": splits["validation"],
            "test": splits["test"],
        }
    )

    captured_kwargs = {}

    def fake_load_dataset(**kwargs):
        captured_kwargs.update(kwargs)
        return stub_dict

    monkeypatch.setattr(dataset_loader, "load_dataset", fake_load_dataset)

    config = DatasetConfig(
        id="irlab-udc/redsm5",
        revision="main",
        splits={"train": "train", "validation": "validation", "test": "test"},
        streaming=False,
        cache_dir="/tmp/cache",
    )
    mlflow_client = DummyMlflow()

    loader = DatasetLoader(mlflow_client=mlflow_client)
    result = loader.load(config)

    assert isinstance(result, StubDatasetDict)
    assert result["train"] is splits["train"]
    assert captured_kwargs == {
        "path": "irlab-udc/redsm5",
        "streaming": False,
        "revision": "main",
        "cache_dir": "/tmp/cache",
    }
    assert mlflow_client.tags["dataset.id"] == "irlab-udc/redsm5"
    assert mlflow_client.tags["dataset.revision"] == "main"
    assert mlflow_client.tags["dataset.resolved_fingerprint"] == "stub-hash"


@pytest.mark.unit
def test_invalid_dataset_id() -> None:
    config = DatasetConfig(id="invalid-id-without-owner")
    loader = DatasetLoader()

    with pytest.raises(DatasetConfigurationError) as err:
        loader.load(config)

    assert "Dataset id 'invalid-id-without-owner' is not valid" in str(err.value)


@pytest.mark.unit
def test_missing_required_split(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_dict = StubDatasetDict({"train": StubSplit(["1"]), "test": StubSplit(["2"])})
    monkeypatch.setattr(dataset_loader, "load_dataset", lambda **_: stub_dict)

    config = DatasetConfig(id="owner/name")

    loader = DatasetLoader()
    with pytest.raises(DatasetConfigurationError) as err:
        loader.load(config)

    message = str(err.value)
    assert "Required dataset splits are missing." in message
    assert "validation" in message


@pytest.mark.unit
def test_split_overlap_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_dict = StubDatasetDict(
        {
            "train": StubSplit(["1", "2"]),
            "validation": StubSplit(["2"]),  # overlap
            "test": StubSplit(["3"]),
        }
    )
    monkeypatch.setattr(dataset_loader, "load_dataset", lambda **_: stub_dict)

    loader = DatasetLoader()
    with pytest.raises(DatasetConfigurationError) as err:
        loader.load(DatasetConfig(id="owner/name"))

    assert "not disjoint" in str(err.value)


@pytest.mark.unit
def test_missing_post_id_column(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoPostIdSplit(StubSplit):
        def __init__(self) -> None:
            super().__init__(["1"])
            self.column_names = ["text"]

    stub_dict = StubDatasetDict(
        {
            "train": NoPostIdSplit(),
            "validation": StubSplit(["2"]),
            "test": StubSplit(["3"]),
        }
    )
    monkeypatch.setattr(dataset_loader, "load_dataset", lambda **_: stub_dict)

    loader = DatasetLoader()
    with pytest.raises(DatasetConfigurationError) as err:
        loader.load(DatasetConfig(id="owner/name"))

    assert "must include a 'post_id' column" in str(err.value)


@pytest.mark.unit
def test_streaming_skips_disjointness(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_dict = StubIterableDatasetDict(
        {
            "train": StubSplit(["1", "2"]),
            "validation": StubSplit(["2"]),  # would overlap if validated
            "test": StubSplit(["3"]),
        }
    )

    def fake_load_dataset(**_):
        return stub_dict

    monkeypatch.setattr(dataset_loader, "load_dataset", fake_load_dataset)

    loader = DatasetLoader()
    config = DatasetConfig(id="owner/name", streaming=True)
    # Should not raise despite overlapping post_ids
    result = loader.load(config)

    assert isinstance(result, StubIterableDatasetDict)


@pytest.mark.unit
def test_load_local_dataset_success(tmp_path: Path) -> None:
    criteria = [
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
    ]

    posts_file = tmp_path / "posts.csv"
    groundtruth_file = tmp_path / "groundtruth.csv"
    annotations_file = tmp_path / "annotations.csv"

    with posts_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["post_id", "text"])
        for idx in range(1, 13):
            writer.writerow([f"p{idx}", f"Post {idx} text"])

    with groundtruth_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["post_id", "text", *criteria])
        for idx in range(1, 13):
            labels = [1 if i == (idx % len(criteria)) else 0 for i in range(len(criteria))]
            writer.writerow([f"p{idx}", f"Post {idx} text", *labels])

    with annotations_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["post_id", "sentence_id", "sentence_text", "DSM5_symptom", "status", "explanation"])
        writer.writerow(["p1", "s1", "Sentence one", "DEPRESSED_MOOD", 1, "Explanation one"])
        writer.writerow(["p2", "s2", "Sentence two", "ANHEDONIA", 0, "Explanation two"])

    mlflow_client = DummyMlflow()
    loader = DatasetLoader(mlflow_client=mlflow_client)
    config = DatasetConfig(
        id="local/redsm5",
        splits={"train": "train", "validation": "validation", "test": "test"},
        local_data={
            "posts_file": str(posts_file),
            "annotations_file": str(annotations_file),
            "groundtruth_file": str(groundtruth_file),
            "split_seed": 2025,
            "split_ratios": {"train": 0.5, "validation": 0.25, "test": 0.25},
        },
    )

    result = loader.load(config)

    assert isinstance(result, StubDatasetDict)

    observed_post_ids = set()
    for split_name in ("train", "validation", "test"):
        assert split_name in result
        split = result[split_name]
        assert isinstance(split, StubSplit)
        ids = split["post_id"]
        assert ids
        observed_post_ids.update(ids)

    expected_post_ids = {f"p{idx}" for idx in range(1, 13)}
    assert observed_post_ids == expected_post_ids
    assert mlflow_client.tags["dataset.id"] == "local/redsm5"
    assert mlflow_client.tags["dataset.revision"] == "local"
