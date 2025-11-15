import optuna

from src.dataaug_multi_both.hpo.search_space import FORBIDDEN_PARAMS, suggest


def test_hpo_param_names_are_clean():
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        params = suggest(trial)
        for k in params:
            assert not any(k.startswith(fp) for fp in FORBIDDEN_PARAMS), k
        for k in params:
            assert (
                k.startswith(
                    (
                        "head.",
                        "loss.",
                        "pred.",
                        "optim.",
                        "sched.",
                        "train.grad_clip_norm",
                        "aug.",
                        "pooling",
                    )
                )
                or k == "pooling"
            ), k
        return 0.0

    study.optimize(objective, n_trials=3)
