import optuna
import pytest
from dataaug_multi_both.hpo import OptunaHPOOptimizer


@pytest.mark.integration
def test_conditional_sampling_smoke():
    # Minimal optimizer just to access _sample_hyperparameters
    opt = OptunaHPOOptimizer(
        study_name="test_smoke",
        storage="sqlite:///:memory:",
    )

    # Define a small search space with conditionals
    search_space = {
        "loss_function": {"type": "categorical", "choices": ["focal", "bce", "hybrid"]},
        "focal_gamma": {"type": "float", "low": 1.0, "high": 3.0},
        "hybrid_weight_alpha": {"type": "float", "low": 0.1, "high": 0.9},
        "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-4},
        "epochs": {"type": "int", "low": 2, "high": 3},
    }

    # Use a real Trial by running a tiny optimize with a custom objective
    def objective(trial: optuna.Trial) -> float:
        params = opt._sample_hyperparameters(trial, search_space)
        loss_fn = params["loss_function"]
        if loss_fn == "focal":
            assert "focal_gamma" in params
        elif loss_fn == "hybrid":
            assert "hybrid_weight_alpha" in params
        else:
            assert "focal_gamma" not in params
            assert "hybrid_weight_alpha" not in params
        # Also ensure loguniform/ints are sampled
        assert 1e-5 <= params["learning_rate"] <= 1e-4
        assert 2 <= params["epochs"] <= 3
        return 0.0

    study = opt.create_or_load_study(load_if_exists=False)
    study.optimize(objective, n_trials=3)

    # If we reached here, sampling worked without raising and assertions passed
    assert len(study.trials) == 3
