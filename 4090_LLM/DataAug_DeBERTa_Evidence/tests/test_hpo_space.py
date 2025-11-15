from __future__ import annotations

import optuna

from dataaug_multi_both.hpo.space import define_search_space


def test_define_search_space_conditionals(test_config):
    study = optuna.create_study(direction="maximize")
    for _ in range(5):
        trial = study.ask()
        cfg = define_search_space(trial, base_config=test_config)
        assert cfg["encoder"]["model_name"] == test_config["encoder"]["model_name"]
        for key in ("evidence", "criteria"):
            head_cfg = cfg["heads"][key]
            if head_cfg["type"] == "mlp":
                assert "hidden" in head_cfg and "layers" in head_cfg
            else:
                assert "hidden" not in head_cfg
                assert "layers" not in head_cfg
            if head_cfg["pooler_type"] == "attention":
                assert "attn_dim" in head_cfg
            else:
                assert "attn_dim" not in head_cfg
        simple_enabled = [
            method
            for method, active in cfg["augmentation"]["simple"]["enabled_mask"].items()
            if active
        ]
        attack_enabled = [
            method
            for method, active in cfg["augmentation"]["ta"]["enabled_mask"].items()
            if active
        ]
        assert len(simple_enabled) <= 2
        assert len(attack_enabled) <= 1
        if simple_enabled:
            assert cfg["augmentation"]["simple"]["compose"] in {"one_of", "sequence"}
        else:
            assert cfg["augmentation"]["simple"]["compose"] == "one_of"
        if not attack_enabled:
            assert cfg["augmentation"]["compose_cross_family"] == "serial_only"
        study.tell(trial, 0.0)
    assert study.best_trial is not None
