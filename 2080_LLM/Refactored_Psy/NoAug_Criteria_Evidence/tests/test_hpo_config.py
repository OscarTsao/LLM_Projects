"""Tests for HPO configuration and setup."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf


def test_hpo_stage_configs_exist():
    """Test that all HPO stage configs exist."""
    config_dir = Path("/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/configs/hpo")
    
    required_configs = [
        "stage0_sanity.yaml",
        "stage1_coarse.yaml",
        "stage2_fine.yaml",
        "stage3_refit.yaml",
    ]
    
    for config_file in required_configs:
        config_path = config_dir / config_file
        assert config_path.exists(), f"HPO config {config_file} should exist"


def test_hpo_stage_configs_valid():
    """Test that HPO stage configs are valid YAML."""
    config_dir = Path("/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/configs/hpo")
    
    for config_file in config_dir.glob("*.yaml"):
        config = OmegaConf.load(config_file)
        
        # Check required fields
        assert "stage" in config, f"{config_file.name} should have 'stage' field"
        assert "n_trials" in config, f"{config_file.name} should have 'n_trials' field"
        assert "direction" in config, f"{config_file.name} should have 'direction' field"


def test_hpo_stage0_is_minimal():
    """Test that stage 0 has minimal trials (sanity check)."""
    config_path = Path("/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/configs/hpo/stage0_sanity.yaml")
    config = OmegaConf.load(config_path)
    
    assert config.n_trials <= 5, "Stage 0 should have minimal trials"


def test_hpo_search_spaces_increase():
    """Test that search space complexity increases from stage 1 to 2."""
    stage1_path = Path("/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/configs/hpo/stage1_coarse.yaml")
    stage2_path = Path("/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/configs/hpo/stage2_fine.yaml")
    
    stage1_config = OmegaConf.load(stage1_path)
    stage2_config = OmegaConf.load(stage2_path)
    
    # Stage 2 should have more trials (finer search)
    assert stage2_config.n_trials >= stage1_config.n_trials, \
        "Stage 2 should have at least as many trials as stage 1"
    
    # Stage 2 should have more hyperparameters to tune
    assert len(stage2_config.search_space) >= len(stage1_config.search_space), \
        "Stage 2 should tune at least as many hyperparameters as stage 1"
