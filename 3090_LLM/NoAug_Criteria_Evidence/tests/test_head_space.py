import importlib

import pytest

PARAMS = [
    ("cls", 1, 256, "gelu", 0.0),
    ("mean", 2, 512, "relu", 0.1),
    ("max", 3, 768, "silu", 0.2),
    ("attn", 4, 1024, "gelu", 0.3),
]


@pytest.mark.parametrize("pooling,layers,hidden,act,drop", PARAMS)
def test_head_param_construction(pooling, layers, hidden, act, drop):
    m = importlib.import_module("psy_agents_noaug.architectures.criteria")
    model = m.Model(
        model_name="bert-base-uncased",
        head_cfg={
            "pooling": pooling,
            "layers": layers,
            "hidden": hidden,
            "activation": act,
            "dropout": drop,
        },
        task_cfg={},
    )
    assert model is not None
