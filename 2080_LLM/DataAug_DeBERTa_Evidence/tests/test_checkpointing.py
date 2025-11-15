from __future__ import annotations

from copy import deepcopy

from dataaug_multi_both.training.train_loop import run_training_job


def test_resume_training_matches_continuous_run(test_config):
    cfg_continuous = deepcopy(test_config)
    cfg_continuous["train"]["num_epochs"] = 2
    result_continuous = run_training_job(cfg_continuous)

    cfg_first = deepcopy(test_config)
    cfg_first["train"]["num_epochs"] = 1
    run_training_job(cfg_first)

    cfg_resumed = deepcopy(test_config)
    cfg_resumed["train"]["num_epochs"] = 2
    resumed = run_training_job(cfg_resumed, resume=True)

    metric_name = cfg_resumed["objective"]["primary_metric"]
    assert abs(result_continuous["metrics"][metric_name] - resumed["metrics"][metric_name]) < 1e-6
