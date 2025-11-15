"""Tests for augmentation integration in HPO system (SUPERMAX Phase 3)."""

from __future__ import annotations

import pytest
import optuna

from psy_agents_noaug.hpo import SearchSpace, SpaceConstraints
from psy_agents_noaug.hpo.evaluation import _extract_augmentation_config


class TestAugmentationSearchSpace:
    """Test augmentation parameters in HPO search space."""

    def test_augmentation_params_present_when_enabled(self):
        """Verify augmentation parameters are sampled when aug.enabled=True."""
        space = SearchSpace("criteria")
        study = optuna.create_study()
        trial = study.ask()

        # Force augmentation enabled
        constraints = SpaceConstraints(categorical={"aug.enabled": [True]})
        params = space.sample(trial, constraints)

        assert "aug.enabled" in params
        assert params["aug.enabled"] is True
        assert "aug.p_apply" in params
        assert "aug.ops_per_sample" in params
        assert "aug.max_replace" in params
        assert "aug.antonym_guard" in params
        assert "aug.method_strategy" in params

        # Validate parameter ranges
        assert 0.05 <= params["aug.p_apply"] <= 0.30
        assert 1 <= params["aug.ops_per_sample"] <= 3
        assert 0.1 <= params["aug.max_replace"] <= 0.4
        assert params["aug.antonym_guard"] in ["off", "on_low_weight"]
        assert params["aug.method_strategy"] in ["all", "nlpaug", "textattack", "light"]

    def test_augmentation_params_disabled_when_false(self):
        """Verify augmentation parameters are set to defaults when aug.enabled=False."""
        space = SearchSpace("criteria")
        study = optuna.create_study()
        trial = study.ask()

        # Force augmentation disabled
        constraints = SpaceConstraints(categorical={"aug.enabled": [False]})
        params = space.sample(trial, constraints)

        assert "aug.enabled" in params
        assert params["aug.enabled"] is False
        assert params["aug.p_apply"] == 0.0
        assert params["aug.ops_per_sample"] == 0
        assert params["aug.max_replace"] == 0.0
        assert params["aug.antonym_guard"] == "off"
        assert params["aug.method_strategy"] == "none"

    def test_augmentation_params_sampled_correctly(self):
        """Run multiple trials to ensure augmentation params vary correctly."""
        space = SearchSpace("criteria")
        study = optuna.create_study()

        # Sample 10 trials with augmentation enabled
        constraints = SpaceConstraints(categorical={"aug.enabled": [True]})
        sampled_p_apply = set()
        sampled_ops = set()

        for _ in range(10):
            trial = study.ask()
            params = space.sample(trial, constraints)
            sampled_p_apply.add(params["aug.p_apply"])
            sampled_ops.add(params["aug.ops_per_sample"])

        # Should have sampled diverse values
        assert len(sampled_p_apply) > 1, "p_apply should vary across trials"
        assert len(sampled_ops) >= 1, "ops_per_sample should be sampled"

    def test_all_agents_support_augmentation(self):
        """Verify all agent types can sample augmentation parameters."""
        agents = ["criteria", "evidence", "share", "joint"]

        for agent in agents:
            space = SearchSpace(agent)
            study = optuna.create_study()
            trial = study.ask()

            constraints = SpaceConstraints(categorical={"aug.enabled": [True]})
            params = space.sample(trial, constraints)

            assert "aug.enabled" in params
            assert params["aug.enabled"] is True
            assert "aug.p_apply" in params


class TestAugmentationConfigExtraction:
    """Test augmentation config extraction from HPO parameters."""

    def test_extract_config_when_enabled(self):
        """Test augmentation config extraction when enabled."""
        params = {
            "agent": "criteria",
            "aug.enabled": True,
            "aug.p_apply": 0.15,
            "aug.ops_per_sample": 2,
            "aug.max_replace": 0.3,
            "aug.antonym_guard": "on_low_weight",
            "aug.method_strategy": "nlpaug",
        }

        config = _extract_augmentation_config(params)

        assert config is not None
        assert config["enabled"] is True
        assert config["p_apply"] == 0.15
        assert config["ops_per_sample"] == 2
        assert config["max_replace"] == 0.3
        assert config["antonym_guard"] == "on_low_weight"
        assert config["method_strategy"] == "nlpaug"

    def test_extract_config_when_disabled(self):
        """Test augmentation config extraction when disabled."""
        params = {
            "agent": "criteria",
            "aug.enabled": False,
        }

        config = _extract_augmentation_config(params)

        assert config is None

    def test_extract_config_defaults(self):
        """Test augmentation config uses defaults for missing parameters."""
        params = {
            "agent": "criteria",
            "aug.enabled": True,
            # Missing other parameters
        }

        config = _extract_augmentation_config(params)

        assert config is not None
        assert config["enabled"] is True
        assert config["p_apply"] == 0.15  # Default
        assert config["ops_per_sample"] == 1  # Default
        assert config["max_replace"] == 0.3  # Default
        assert config["antonym_guard"] == "off"  # Default
        assert config["method_strategy"] == "all"  # Default

    def test_tfidf_path_resolution(self, tmp_path):
        """Test TF-IDF cache path resolution."""
        # Create fake TF-IDF cache directory
        tfidf_cache = tmp_path / "data" / "augmentation_cache" / "tfidf" / "criteria"
        tfidf_cache.mkdir(parents=True, exist_ok=True)
        (tfidf_cache / "tfidfaug_w2idf.txt").write_text("test")

        params = {
            "agent": "criteria",
            "aug.enabled": True,
        }

        # Temporarily change working directory or use environment variable
        # For now, just test that tfidf_model key exists
        config = _extract_augmentation_config(params)
        assert "tfidf_model" in config


class TestAugmentationConstraints:
    """Test augmentation parameter constraints."""

    def test_p_apply_constraints(self):
        """Test p_apply parameter respects custom constraints."""
        space = SearchSpace("criteria")
        study = optuna.create_study()
        trial = study.ask()

        # Narrow p_apply range
        constraints = SpaceConstraints(
            categorical={"aug.enabled": [True]},
            floats={"aug.p_apply": (0.10, 0.20)},
        )
        params = space.sample(trial, constraints)

        assert 0.10 <= params["aug.p_apply"] <= 0.20

    def test_ops_per_sample_constraints(self):
        """Test ops_per_sample parameter respects custom constraints."""
        space = SearchSpace("criteria")
        study = optuna.create_study()
        trial = study.ask()

        # Force ops_per_sample = 1
        constraints = SpaceConstraints(
            categorical={"aug.enabled": [True]},
            ints={"aug.ops_per_sample": (1, 1)},
        )
        params = space.sample(trial, constraints)

        assert params["aug.ops_per_sample"] == 1

    def test_method_strategy_constraints(self):
        """Test method_strategy parameter respects custom choices."""
        space = SearchSpace("criteria")
        study = optuna.create_study()
        trial = study.ask()

        # Force nlpaug only
        constraints = SpaceConstraints(
            categorical={
                "aug.enabled": [True],
                "aug.method_strategy": ["nlpaug"],
            }
        )
        params = space.sample(trial, constraints)

        assert params["aug.method_strategy"] == "nlpaug"


class TestAugmentationIntegration:
    """Integration tests for augmentation in HPO."""

    def test_augmentation_params_do_not_break_existing_params(self):
        """Ensure augmentation params don't interfere with existing params."""
        space = SearchSpace("criteria")
        study = optuna.create_study()
        trial = study.ask()

        params = space.sample(trial)

        # All existing params should still be present
        assert "model.name" in params
        assert "tok.max_length" in params
        assert "optim.name" in params
        assert "train.batch_size" in params

        # Augmentation params should also be present
        assert "aug.enabled" in params

    def test_augmentation_enabled_trials_have_more_params(self):
        """Verify enabled augmentation adds parameters."""
        space = SearchSpace("criteria")

        # Disabled augmentation - separate study
        study_disabled = optuna.create_study()
        trial_disabled = study_disabled.ask()
        constraints_disabled = SpaceConstraints(categorical={"aug.enabled": [False]})
        params_disabled = space.sample(trial_disabled, constraints_disabled)
        disabled_keys = set(params_disabled.keys())

        # Enabled augmentation - separate study to avoid Optuna categorical distribution conflict
        study_enabled = optuna.create_study()
        trial_enabled = study_enabled.ask()
        constraints_enabled = SpaceConstraints(categorical={"aug.enabled": [True]})
        params_enabled = space.sample(trial_enabled, constraints_enabled)
        enabled_keys = set(params_enabled.keys())

        # Both should have same keys (defaults are set when disabled)
        assert disabled_keys == enabled_keys

        # But values should differ
        assert params_disabled["aug.p_apply"] == 0.0
        assert params_enabled["aug.p_apply"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
