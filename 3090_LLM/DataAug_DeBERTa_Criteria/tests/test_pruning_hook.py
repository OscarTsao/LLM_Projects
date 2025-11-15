import optuna


def test_pruning_flow():
    def obj(trial):
        trial.report(0.1, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned
        trial.report(0.0, step=2)
        raise optuna.TrialPruned

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=1, catch=(optuna.TrialPruned,))
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
