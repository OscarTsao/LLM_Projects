import torch

from src.dataaug_multi_both.training.losses import build_criterion


def test_losses_grad():
    logits = torch.randn(3, 5, requires_grad=True)
    targets = torch.randint(0, 2, (3,5)).float()
    for name in ["bce","weighted_bce","focal","asymmetric"]:
        posw = torch.ones(5) if name in ("weighted_bce",) else None
        crit = build_criterion({"loss.name": name, "loss.focal.gamma":2.0,"loss.focal.alpha_pos":0.25,"loss.asym.gamma_pos":0.0,"loss.asym.gamma_neg":4.0}, posw)
        loss = crit(logits, targets)
        assert torch.isfinite(loss).item()
        loss.backward(retain_graph=True)
