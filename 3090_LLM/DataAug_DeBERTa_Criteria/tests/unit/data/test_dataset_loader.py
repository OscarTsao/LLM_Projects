from types import SimpleNamespace

import pytest

import src.dataaug_multi_both.data.dataset_loader as dataset_loader
from src.dataaug_multi_both.data.dataset_loader import (
    DatasetConfig,
    DatasetConfigurationError,
    DatasetLoader,
)


class StubSplit:
    def __init__(self, post_ids: list[str], fingerprint: str = "stub-hash") -> None:
        self._post_ids = post_ids
        self._fingerprint = fingerprint
        self.column_names = ["post_id", "text"]
        self.info = SimpleNamespace(hash="info-hash")

    def __getitem__(self, key: str) -> list[str]:
        if key == "post_id":
            return self._post_ids
        raise KeyError(key)


class StubDatasetDict(dict):
    """Minimal stand-in for datasets.DatasetDict."""


class StubIterableDatasetDict(dict):
    """Minimal stand-in for datasets.IterableDatasetDict."""


class DummyMlflow:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}

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
