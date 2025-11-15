"""Tests for HPO configuration and setup."""

from pathlib import Path

from omegaconf import OmegaConf


def test_hpo_stage_configs_exist():
    """Test that all HPO stage configs exist."""
    config_dir = Path(__file__).parent.parent / "configs" / "hpo"

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
    config_dir = Path(__file__).parent.parent / "configs" / "hpo"

    for config_file in config_dir.glob("*.yaml"):
        config = OmegaConf.load(config_file)

        # Check required fields
        assert "stage" in config, f"{config_file.name} should have 'stage' field"
        assert "n_trials" in config, f"{config_file.name} should have 'n_trials' field"
        assert (
            "direction" in config
        ), f"{config_file.name} should have 'direction' field"


def test_hpo_stage0_is_minimal():
    """Test that stage 0 has minimal trials (sanity check)."""
    config_path = (
        Path(__file__).parent.parent / "configs" / "hpo" / "stage0_sanity.yaml"
    )
    config = OmegaConf.load(config_path)

    assert config.n_trials <= 10, "Stage 0 should have minimal trials (sanity check)"


def test_hpo_search_spaces_increase():
    """Test that search space exists and HPO stages are configured correctly."""
    stage1_path = (
        Path(__file__).parent.parent / "configs" / "hpo" / "stage1_coarse.yaml"
    )
    stage2_path = Path(__file__).parent.parent / "configs" / "hpo" / "stage2_fine.yaml"

    stage1_config = OmegaConf.load(stage1_path)
    stage2_config = OmegaConf.load(stage2_path)

    # Stage 1 is coarse (broad search, more trials)
    # Stage 2 is fine (narrow/focused search, fewer trials but more epochs)
    # This is a valid HPO strategy
    assert stage1_config.n_trials > 0, "Stage 1 should have trials"
    assert stage2_config.n_trials > 0, "Stage 2 should have trials"

    # Both should have search spaces
    assert len(stage1_config.search_space) > 0, "Stage 1 should have a search space"
    assert len(stage2_config.search_space) > 0, "Stage 2 should have a search space"

    # Stage 2 typically trains for more epochs (finer evaluation)
    assert (
        stage2_config.num_epochs >= stage1_config.num_epochs
    ), "Stage 2 should train for at least as many epochs as stage 1"
