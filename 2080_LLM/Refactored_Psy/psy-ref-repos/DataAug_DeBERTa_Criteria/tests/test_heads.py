import torch

from src.dataaug_multi_both.models.heads.criteria_matching import build_criteria_head


def _run(htype):
    cfg = {"head.type": htype, "head.dropout": 0.1}
    if htype == "mlp":
        cfg.update({"head.mlp.layers": 2, "head.mlp.hidden_dim": 512, "head.mlp.activation": "gelu", "head.mlp.layernorm": True})
    if htype == "glu":
        cfg.update({"head.glu.hidden_dim": 384, "head.glu.gate_bias": True})
    if htype == "msd":
        cfg.update({"head.msd.n_samples": 4, "head.msd.alpha": 0.5})
    head = build_criteria_head(cfg, hidden_size=768, num_labels=8)
    x = torch.randn(4, 768)
    y = head(x)
    assert y.shape == (4, 8)

def test_all_heads():
    for t in ["linear", "mlp", "glu", "msd"]:
        _run(t)
