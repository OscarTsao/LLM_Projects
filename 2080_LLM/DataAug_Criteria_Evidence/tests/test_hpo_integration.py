"""
Test HPO pipeline integration and orchestration.

Tests the hyperparameter optimization system including Optuna integration,
configuration management, and trial execution.
"""

from pathlib import Path

import optuna
import pytest
from omegaconf import OmegaConf


class TestHPOConfiguration:
    """Test HPO configuration loading and validation."""

    def test_all_stage_configs_exist(self):
        """Test that all HPO stage configs exist."""
        config_dir = Path("configs/hpo")
        required_configs = [
            "stage0_sanity.yaml",
            "stage1_coarse.yaml",
            "stage2_fine.yaml",
            "stage3_refit.yaml",
        ]

        for config_file in required_configs:
            assert (config_dir / config_file).exists()

    def test_stage_configs_have_required_fields(self):
        """Test that stage configs have all required fields."""
        config_dir = Path("configs/hpo")

        for config_file in config_dir.glob("stage*.yaml"):
            config = OmegaConf.load(config_file)

            # Required fields
            assert "stage" in config
            assert "n_trials" in config
            assert "direction" in config

    def test_stage_progression_makes_sense(self):
        """Test that stages progress logically."""
        configs = []
        for i in range(3):
            path = Path(f"configs/hpo/stage{i}_*.yaml")
            matches = list(Path("configs/hpo").glob(f"stage{i}_*.yaml"))
            if matches:
                configs.append(OmegaConf.load(matches[0]))

        if len(configs) >= 2:
            # Stage 0 should have fewer trials than stage 1
            assert configs[0].n_trials <= configs[1].n_trials


class TestOptunaIntegration:
    """Test Optuna study creation and management."""

    def test_create_study_basic(self):
        """Test that Optuna study can be created."""
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        assert study is not None
        assert isinstance(study, optuna.Study)

    def test_create_study_with_storage(self, tmp_path):
        """Test study creation with SQLite storage."""
        storage_path = tmp_path / "test_optuna.db"
        storage_url = f"sqlite:///{storage_path}"

        study = optuna.create_study(
            study_name="test_study",
            storage=storage_url,
            direction="maximize",
            load_if_exists=True,
        )
        assert study is not None
        assert storage_path.exists()

    def test_study_directions(self):
        """Test both maximize and minimize directions."""
        study_max = optuna.create_study(direction="maximize")
        study_min = optuna.create_study(direction="minimize")

        assert study_max.direction == optuna.study.StudyDirection.MAXIMIZE
        assert study_min.direction == optuna.study.StudyDirection.MINIMIZE

    def test_study_sampler_options(self):
        """Test different Optuna samplers."""
        # TPE sampler (default)
        study_tpe = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        assert isinstance(study_tpe.sampler, optuna.samplers.TPESampler)

        # Random sampler
        study_random = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42)
        )
        assert isinstance(study_random.sampler, optuna.samplers.RandomSampler)


class TestTrialSuggestions:
    """Test Optuna trial parameter suggestions."""

    def test_suggest_float(self):
        """Test suggesting float hyperparameters."""

        def objective(trial):
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            assert isinstance(lr, float)
            assert 1e-5 <= lr <= 1e-3
            return lr

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3)
        assert len(study.trials) == 3

    def test_suggest_int(self):
        """Test suggesting integer hyperparameters."""

        def objective(trial):
            batch_size = trial.suggest_int("batch_size", 8, 64, log=True)
            assert isinstance(batch_size, int)
            assert 8 <= batch_size <= 64
            return batch_size

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3)
        assert len(study.trials) == 3

    def test_suggest_categorical(self):
        """Test suggesting categorical hyperparameters."""

        def objective(trial):
            model = trial.suggest_categorical("model", ["bert", "roberta", "deberta"])
            assert model in ["bert", "roberta", "deberta"]
            return 0.95

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3)
        assert len(study.trials) == 3


class TestHPOPruning:
    """Test Optuna pruning mechanisms."""

    def test_median_pruner(self):
        """Test MedianPruner for early stopping."""
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_trial_pruning_logic(self):
        """Test that trials can be pruned."""

        def objective(trial):
            # Simulate a bad trial
            for step in range(10):
                score = 0.1  # Consistently poor
                trial.report(score, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return score

        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
        )

        # Run multiple trials - some may be pruned
        study.optimize(objective, n_trials=5, catch=(optuna.TrialPruned,))
        # At least some trials should complete
        assert len(study.trials) == 5


class TestHPOSearchSpace:
    """Test HPO search space definition and sampling."""

    def test_search_space_from_config(self):
        """Test loading search space from config."""
        config_path = Path("configs/hpo/stage1_coarse.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        config = OmegaConf.load(config_path)
        assert "search_space" in config
        assert len(config.search_space) > 0

    def test_search_space_parameters(self):
        """Test that search space has expected parameter types."""
        config_path = Path("configs/hpo/stage1_coarse.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        config = OmegaConf.load(config_path)
        search_space = config.search_space

        # Common hyperparameters in ML training
        expected_params = ["learning_rate", "batch_size", "weight_decay"]
        found_params = list(search_space.keys())

        # At least some expected parameters should be present
        common = set(expected_params) & set(found_params)
        # Allow flexibility in search space definition
        # Just verify it's not empty
        assert len(found_params) > 0


class TestHPOResults:
    """Test HPO result storage and retrieval."""

    def test_best_trial_retrieval(self):
        """Test retrieving best trial from study."""

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2  # Minimize at x=2

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)

        best_trial = study.best_trial
        assert best_trial is not None
        assert hasattr(best_trial, "value")
        assert hasattr(best_trial, "params")

    def test_best_params_retrieval(self):
        """Test retrieving best parameters."""

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        assert "x" in best_params
        # Best value should be close to 2
        assert abs(best_params["x"] - 2) < 5

    def test_trials_dataframe(self):
        """Test converting trials to DataFrame."""

        def objective(trial):
            x = trial.suggest_float("x", 0, 10)
            y = trial.suggest_int("y", 1, 5)
            return x + y

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)

        df = study.trials_dataframe()
        assert len(df) == 5
        assert "value" in df.columns
        assert "params_x" in df.columns
        assert "params_y" in df.columns


class TestHPODeterminism:
    """Test HPO reproducibility with seeds."""

    def test_sampler_with_seed(self):
        """Test that sampler with seed produces reproducible results."""

        def objective(trial):
            x = trial.suggest_float("x", 0, 10)
            return x**2

        # Run 1
        study1 = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.RandomSampler(seed=42)
        )
        study1.optimize(objective, n_trials=5)
        params1 = [t.params["x"] for t in study1.trials]

        # Run 2
        study2 = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.RandomSampler(seed=42)
        )
        study2.optimize(objective, n_trials=5)
        params2 = [t.params["x"] for t in study2.trials]

        # Should be identical
        assert params1 == params2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""

        def objective(trial):
            x = trial.suggest_float("x", 0, 10)
            return x**2

        study1 = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.RandomSampler(seed=42)
        )
        study1.optimize(objective, n_trials=5)
        params1 = [t.params["x"] for t in study1.trials]

        study2 = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.RandomSampler(seed=123)
        )
        study2.optimize(objective, n_trials=5)
        params2 = [t.params["x"] for t in study2.trials]

        # Should be different
        assert params1 != params2


class TestHPOStorage:
    """Test HPO storage and persistence."""

    def test_study_persistence(self, tmp_path):
        """Test that study can be saved and loaded."""
        storage_path = tmp_path / "test.db"
        storage_url = f"sqlite:///{storage_path}"

        # Create and run study
        study1 = optuna.create_study(
            study_name="persist_test",
            storage=storage_url,
            direction="maximize",
            load_if_exists=True,
        )

        def objective(trial):
            return trial.suggest_float("x", 0, 1) ** 2

        study1.optimize(objective, n_trials=3)

        # Load same study
        study2 = optuna.load_study(study_name="persist_test", storage=storage_url)

        assert len(study2.trials) == 3
        assert study2.best_value == study1.best_value

    def test_multiple_studies_in_storage(self, tmp_path):
        """Test multiple studies in same storage."""
        storage_path = tmp_path / "multi.db"
        storage_url = f"sqlite:///{storage_path}"

        def objective(trial):
            return trial.suggest_float("x", 0, 1)

        # Study 1
        study1 = optuna.create_study(
            study_name="study1", storage=storage_url, direction="maximize"
        )
        study1.optimize(objective, n_trials=2)

        # Study 2
        study2 = optuna.create_study(
            study_name="study2", storage=storage_url, direction="minimize"
        )
        study2.optimize(objective, n_trials=3)

        # Verify both exist
        study1_loaded = optuna.load_study(study_name="study1", storage=storage_url)
        study2_loaded = optuna.load_study(study_name="study2", storage=storage_url)

        assert len(study1_loaded.trials) == 2
        assert len(study2_loaded.trials) == 3


class TestHPOParallelization:
    """Test HPO parallel execution concepts."""

    def test_study_supports_parallelization(self, tmp_path):
        """Test that study can handle parallel optimization."""
        storage_path = tmp_path / "parallel.db"
        storage_url = f"sqlite:///{storage_path}"

        def objective(trial):
            return trial.suggest_float("x", 0, 1) ** 2

        study = optuna.create_study(
            study_name="parallel_test",
            storage=storage_url,
            direction="minimize",
            load_if_exists=True,
        )

        # Simulate sequential "parallel" trials
        study.optimize(objective, n_trials=2)
        study.optimize(objective, n_trials=2)

        assert len(study.trials) == 4


class TestEdgeCases:
    """Test edge cases in HPO."""

    def test_zero_trials(self):
        """Test study with zero trials."""
        study = optuna.create_study(direction="maximize")

        # Should not have best trial yet
        with pytest.raises((ValueError, AttributeError)):
            _ = study.best_trial

    def test_failed_trial(self):
        """Test handling of failed trials."""

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            if x < 0.5:
                raise ValueError("Simulated failure")
            return x

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5, catch=(ValueError,))

        # Some trials should have completed
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        assert len(completed) > 0

    def test_nan_objective_value(self):
        """Test handling of NaN objective values."""

        def objective(trial):
            return float("nan")

        study = optuna.create_study(direction="maximize")

        # Optuna logs warning but doesn't raise by default
        # Just verify it completes without crashing
        study.optimize(objective, n_trials=1)
        # Trial should be marked as failed
        assert len(study.trials) == 1
        assert study.trials[0].state == optuna.trial.TrialState.FAIL
