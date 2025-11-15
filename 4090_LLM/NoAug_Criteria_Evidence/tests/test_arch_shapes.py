import importlib

import torch
from transformers import AutoTokenizer

ARCHS = {
    "criteria": "psy_agents_noaug.architectures.criteria",
    "evidence": "psy_agents_noaug.architectures.evidence",
    "share": "psy_agents_noaug.architectures.share",
    "joint": "psy_agents_noaug.architectures.joint",
}


def _tok():
    t = AutoTokenizer.from_pretrained("bert-base-uncased")
    return t("this is a test input", return_tensors="pt")


def test_forward_shapes():
    inputs = _tok()
    for name, path in ARCHS.items():
        m = importlib.import_module(path)
        # The Model constructor must accept (model_name, head_cfg, task_cfg)
        model = m.Model(
            model_name="bert-base-uncased",
            head_cfg={
                "layers": 1,
                "hidden": 256,
                "activation": "gelu",
                "dropout": 0.1,
                "pooling": "cls",
            },
            task_cfg={},
        )
        model.eval()
        with torch.no_grad():
            out = model(**inputs) if name != "evidence" else model(inputs)
        if name in ("criteria", "share", "joint"):
            logits = getattr(out, "logits", out["logits"])
            assert logits.ndim == 2 and logits.shape[0] == 1
        if name in ("evidence", "share", "joint"):
            s = (
                getattr(
                    out,
                    "start_logits",
                    out.get("start_logits", None) if isinstance(out, dict) else None,
                )
                or out["start_logits"]
            )
            e = (
                getattr(
                    out,
                    "end_logits",
                    out.get("end_logits", None) if isinstance(out, dict) else None,
                )
                or out["end_logits"]
            )
            assert s.shape == e.shape and s.shape[0] == 1
