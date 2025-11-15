import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.dataaug_multi_both.data.dataset_loader import (
    Dataset,
    DatasetConfig,
    DatasetConfigurationError,
    DatasetDict,
    DatasetLoader,
    IterableDatasetDict,
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


class DummyMlflow:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Write CSV data
        f.write("post_id,sentence_id,sentence_text,DSM5_symptom,status,explanation\n")
        f.write('1,1_1,"Test sentence 1",DEPRESSED_MOOD,1,"Test explanation 1"\n')
        f.write('2,2_1,"Test sentence 2",FATIGUE,1,"Test explanation 2"\n')
        f.write('3,3_1,"Test sentence 3",ANHEDONIA,0,"Test explanation 3"\n')
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.mark.unit
def test_load_dataset_success(temp_csv_file: Path) -> None:
    config = DatasetConfig(
        id="csv",
        data_files=str(temp_csv_file),
        splits={"train": "train", "validation": "validation", "test": "test"},
        streaming=False,
        split_percentages={"train": 0.4, "validation": 0.3, "test": 0.3},
    )
    mlflow_client = DummyMlflow()

    loader = DatasetLoader(mlflow_client=mlflow_client)
    result = loader.load(config)

    assert isinstance(result, DatasetDict)
    assert "train" in result
    assert "validation" in result
    assert "test" in result
    assert mlflow_client.tags["dataset.id"] == "csv"


@pytest.mark.unit
def test_invalid_dataset_id() -> None:
    config = DatasetConfig(id="invalid-id", data_files="test.csv")
    loader = DatasetLoader()

    with pytest.raises(DatasetConfigurationError) as err:
        loader.load(config)

    assert "must be a local dataset builder" in str(err.value)


@pytest.mark.unit
def test_missing_data_files() -> None:
    config = DatasetConfig(id="csv")
    loader = DatasetLoader()

    with pytest.raises(DatasetConfigurationError) as err:
        loader.load(config)

    assert "data_files must be specified" in str(err.value)


@pytest.mark.unit
def test_split_overlap_detected(temp_csv_file: Path) -> None:
    # Create a CSV with overlapping post_ids
    overlap_file = temp_csv_file.parent / "overlap.csv"
    with open(overlap_file, "w") as f:
        f.write("post_id,sentence_id,sentence_text,DSM5_symptom,status,explanation\n")
        f.write('1,1_1,"Test 1",DEPRESSED_MOOD,1,"Exp 1"\n')
        f.write('2,2_1,"Test 2",FATIGUE,1,"Exp 2"\n')
        f.write('3,3_1,"Test 3",ANHEDONIA,0,"Exp 3"\n')

    try:
        # Manually create overlapping splits
        df = pd.read_csv(overlap_file)
        # Reset index to ensure correct filtering
        df = df.reset_index(drop=True)
        stub_dict = DatasetDict(
            {
                "train": Dataset(df[df["post_id"].isin([1, 2])].reset_index(drop=True)),
                "validation": Dataset(df[df["post_id"].isin([2])].reset_index(drop=True)),  # overlap with train
                "test": Dataset(df[df["post_id"].isin([3])].reset_index(drop=True)),
            }
        )

        loader = DatasetLoader()
        with pytest.raises(DatasetConfigurationError) as err:
            loader._assert_split_disjointness(stub_dict, streaming=False)

        assert "not disjoint" in str(err.value)
    finally:
        overlap_file.unlink()


@pytest.mark.unit
def test_missing_post_id_column() -> None:
    # Create a CSV without post_id column
    missing_file = Path(tempfile.mktemp(suffix=".csv"))
    with open(missing_file, "w") as f:
        f.write("sentence_id,sentence_text,DSM5_symptom,status,explanation\n")
        f.write('1_1,"Test 1",DEPRESSED_MOOD,1,"Exp 1"\n')

    try:
        df = pd.read_csv(missing_file)
        ds = Dataset(df)
        stub_dict = DatasetDict({"train": ds, "validation": ds, "test": ds})

        loader = DatasetLoader()
        with pytest.raises(DatasetConfigurationError) as err:
            loader._assert_split_disjointness(stub_dict, streaming=False)

        assert "must include a 'post_id' column" in str(err.value)
    finally:
        missing_file.unlink()


@pytest.mark.unit
def test_streaming_skips_disjointness(temp_csv_file: Path) -> None:
    # Create overlapping splits but with streaming=True
    df = pd.read_csv(temp_csv_file)
    stub_dict = IterableDatasetDict(
        {
            "train": Dataset(df[df["post_id"].isin(["1", "2"])]),
            "validation": Dataset(df[df["post_id"].isin(["2"])]),  # would overlap if validated
            "test": Dataset(df[df["post_id"].isin(["3"])]),
        }
    )

    loader = DatasetLoader()
    # Should not raise despite overlapping post_ids
    loader._assert_split_disjointness(stub_dict, streaming=True)
