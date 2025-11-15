"""Unit tests for dataset loading."""

import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset, DatasetDict
from src.dataaug_multi_both.data.dataset import (
    load_hf_dataset,
    DatasetValidator,
    DatasetLoadError
)


class TestDatasetValidator:
    """Test suite for DatasetValidator."""
    
    def test_validate_dataset_id_valid(self):
        """Test that valid dataset IDs pass validation."""
        assert DatasetValidator.validate_dataset_id("irlab-udc/redsm5")
        assert DatasetValidator.validate_dataset_id("org/dataset")
        assert DatasetValidator.validate_dataset_id("user/my-dataset-123")
    
    def test_validate_dataset_id_invalid_format(self):
        """Test that invalid dataset ID format raises error."""
        with pytest.raises(DatasetLoadError, match="Invalid dataset identifier format"):
            DatasetValidator.validate_dataset_id("invalid_dataset")
    
    def test_validate_dataset_id_empty(self):
        """Test that empty dataset ID raises error."""
        with pytest.raises(DatasetLoadError, match="Invalid dataset identifier"):
            DatasetValidator.validate_dataset_id("")
    
    def test_validate_dataset_id_none(self):
        """Test that None dataset ID raises error."""
        with pytest.raises(DatasetLoadError, match="Invalid dataset identifier"):
            DatasetValidator.validate_dataset_id(None)
    
    def test_validate_splits_all_present(self):
        """Test that validation passes when all splits present."""
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"text": ["a", "b"]}),
            "validation": Dataset.from_dict({"text": ["c"]}),
            "test": Dataset.from_dict({"text": ["d"]})
        })
        
        assert DatasetValidator.validate_splits(mock_dataset)
    
    def test_validate_splits_missing(self):
        """Test that validation fails when splits missing."""
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"text": ["a", "b"]}),
            "test": Dataset.from_dict({"text": ["d"]})
        })
        
        with pytest.raises(DatasetLoadError, match="Missing required splits.*validation"):
            DatasetValidator.validate_splits(mock_dataset)
    
    def test_validate_split_disjointness_valid(self):
        """Test that disjoint splits pass validation."""
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"post_id": [1, 2, 3]}),
            "validation": Dataset.from_dict({"post_id": [4, 5]}),
            "test": Dataset.from_dict({"post_id": [6, 7]})
        })
        
        assert DatasetValidator.validate_split_disjointness(mock_dataset)
    
    def test_validate_split_disjointness_overlap(self):
        """Test that overlapping splits fail validation."""
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"post_id": [1, 2, 3]}),
            "validation": Dataset.from_dict({"post_id": [3, 4]}),  # 3 overlaps
            "test": Dataset.from_dict({"post_id": [5, 6]})
        })
        
        with pytest.raises(DatasetLoadError, match="Splits are not disjoint"):
            DatasetValidator.validate_split_disjointness(mock_dataset)
    
    def test_validate_split_disjointness_missing_id_column(self):
        """Test that missing ID column is handled gracefully."""
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"text": ["a", "b"]}),
            "validation": Dataset.from_dict({"text": ["c"]})
        })
        
        # Should not raise error, just log warning
        assert DatasetValidator.validate_split_disjointness(mock_dataset)


class TestLoadHFDataset:
    """Test suite for load_hf_dataset function."""
    
    @patch('src.dataaug_multi_both.data.dataset.load_dataset')
    def test_load_dataset_success(self, mock_load):
        """Test successful dataset loading."""
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"post_id": [1, 2], "text": ["a", "b"]}),
            "validation": Dataset.from_dict({"post_id": [3], "text": ["c"]}),
            "test": Dataset.from_dict({"post_id": [4], "text": ["d"]})
        })
        mock_load.return_value = mock_dataset
        
        dataset, metadata = load_hf_dataset("irlab-udc/redsm5")
        
        assert dataset is not None
        assert "train" in dataset
        assert "validation" in dataset
        assert "test" in dataset
        assert metadata["dataset_id"] == "irlab-udc/redsm5"
        assert metadata["revision"] == "main"
        assert "resolved_hash" in metadata
    
    @patch('src.dataaug_multi_both.data.dataset.load_dataset')
    def test_load_dataset_invalid_id(self, mock_load):
        """Test that invalid dataset ID raises error."""
        with pytest.raises(DatasetLoadError, match="Invalid dataset identifier format"):
            load_hf_dataset("invalid_dataset")
    
    @patch('src.dataaug_multi_both.data.dataset.load_dataset')
    def test_load_dataset_missing_splits(self, mock_load):
        """Test that missing splits raise error."""
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"text": ["a", "b"]}),
            # Missing validation and test
        })
        mock_load.return_value = mock_dataset
        
        with pytest.raises(DatasetLoadError, match="Missing required splits"):
            load_hf_dataset("irlab-udc/redsm5")
    
    @patch('src.dataaug_multi_both.data.dataset.load_dataset')
    def test_load_dataset_network_error(self, mock_load):
        """Test that network errors produce actionable error."""
        mock_load.side_effect = ConnectionError("Network unreachable")
        
        with pytest.raises(DatasetLoadError, match="Failed to load dataset.*Remediation"):
            load_hf_dataset("irlab-udc/redsm5")
    
    @patch('src.dataaug_multi_both.data.dataset.load_dataset')
    def test_load_dataset_overlapping_splits(self, mock_load):
        """Test that overlapping splits raise error."""
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"post_id": [1, 2, 3]}),
            "validation": Dataset.from_dict({"post_id": [3, 4]}),  # 3 overlaps
            "test": Dataset.from_dict({"post_id": [5, 6]})
        })
        mock_load.return_value = mock_dataset
        
        with pytest.raises(DatasetLoadError, match="Splits are not disjoint"):
            load_hf_dataset("irlab-udc/redsm5", validate_disjoint=True)
    
    @patch('src.dataaug_multi_both.data.dataset.load_dataset')
    def test_load_dataset_with_revision(self, mock_load):
        """Test loading dataset with specific revision."""
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"post_id": [1, 2], "text": ["a", "b"]}),
            "validation": Dataset.from_dict({"post_id": [3], "text": ["c"]}),
            "test": Dataset.from_dict({"post_id": [4], "text": ["d"]})
        })
        mock_load.return_value = mock_dataset
        
        dataset, metadata = load_hf_dataset("irlab-udc/redsm5", revision="v1.0")
        
        assert metadata["revision"] == "v1.0"
        mock_load.assert_called_once_with(
            "irlab-udc/redsm5",
            revision="v1.0",
            cache_dir=None
        )
    
    @patch('src.dataaug_multi_both.data.dataset.load_dataset')
    def test_load_dataset_metadata_includes_stats(self, mock_load):
        """Test that metadata includes dataset statistics."""
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"post_id": [1, 2, 3], "text": ["a", "b", "c"]}),
            "validation": Dataset.from_dict({"post_id": [4], "text": ["d"]}),
            "test": Dataset.from_dict({"post_id": [5, 6], "text": ["e", "f"]})
        })
        mock_load.return_value = mock_dataset
        
        dataset, metadata = load_hf_dataset("irlab-udc/redsm5")
        
        assert "num_examples" in metadata
        assert metadata["num_examples"]["train"] == 3
        assert metadata["num_examples"]["validation"] == 1
        assert metadata["num_examples"]["test"] == 2
        assert "splits" in metadata
        assert set(metadata["splits"]) == {"train", "validation", "test"}

