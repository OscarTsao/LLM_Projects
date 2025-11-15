from __future__ import annotations

from copy import deepcopy

from dataaug_multi_both.augment.textattack_factory import build_augmenter


def test_augmentation_applies_only_to_evidence_field():
    cfg = {
        "seed": 123,
        "apply_to": "evidence",
        "simple": {
            "compose": "one_of",
            "enabled_mask": {
                "EDA": True,
                "CharSwap": False,
                "Embedding": False,
                "BackTranslation": False,
                "CheckList": False,
                "CLARE": False,
            },
            "params": {
                "EDA": {"p_token": 0.2, "tpe": 1},
                "CharSwap": {"p_token": 0.2, "tpe": 1},
                "Embedding": {"p_token": 0.2, "tpe": 1},
                "BackTranslation": {"p_token": 0.2, "tpe": 1},
                "CheckList": {"p_token": 0.2, "tpe": 1},
                "CLARE": {"p_token": 0.2, "tpe": 1},
            },
        },
        "ta": {
            "enabled_mask": {name: False for name in [
                "TextFoolerJin2019",
                "PWWSRen2019",
                "DeepWordBugGao2018",
                "HotFlipEbrahimi2017",
                "IGAWang2019",
                "Kuleshov2017",
                "CheckList2020",
                "BAEGarg2019",
            ]},
            "params": {
                name: {"p_token": 0.2, "tpe": 1}
                for name in [
                    "TextFoolerJin2019",
                    "PWWSRen2019",
                    "DeepWordBugGao2018",
                    "HotFlipEbrahimi2017",
                    "IGAWang2019",
                    "Kuleshov2017",
                    "CheckList2020",
                    "BAEGarg2019",
                ]
            },
        },
        "compose_cross_family": "serial_only",
    }
    augmenter = build_augmenter(cfg)

    train_example = {"text": "original text", "evidence": "original evidence"}
    val_example = deepcopy(train_example)

    augmented = deepcopy(train_example)
    augmented["evidence"] = augmenter(augmented["evidence"])

    assert augmented["text"] == train_example["text"]
    assert augmented["evidence"] != train_example["evidence"]
    assert val_example["evidence"] == train_example["evidence"]
