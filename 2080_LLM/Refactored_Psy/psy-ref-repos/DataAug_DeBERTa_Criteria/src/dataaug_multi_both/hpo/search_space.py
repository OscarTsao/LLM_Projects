from __future__ import annotations

import optuna

ALLOWED_PARAM_PREFIXES = (
    "head.", "pooling", "loss.", "pred.", "optim.", "sched.", "train.grad_clip_norm", "aug."
)
FORBIDDEN_PARAMS = {
    "batch_size", "per_device_batch_size", "grad_accumulation_steps",
    "mixed_precision", "amp_dtype", "grad_checkpointing", "num_workers",
    "pin_memory", "max_seq_len", "max_seq_length", "ddp", "world_size", "torch_compile",
}

def suggest(trial: optuna.Trial) -> dict:
    head_type = trial.suggest_categorical("head.type", ["linear", "mlp", "glu", "msd"])
    params = {
        "head.type": head_type,
        "pooling": trial.suggest_categorical("pooling", ["cls", "mean"]),
        "head.dropout": trial.suggest_float("head.dropout", 0.0, 0.5),
    }
    if head_type == "linear":
        params["head.bias"] = trial.suggest_categorical("head.bias", [True, False])
    elif head_type == "mlp":
        params["head.mlp.layers"] = trial.suggest_int("head.mlp.layers", 1, 3)
        params["head.mlp.hidden_dim"] = trial.suggest_categorical("head.mlp.hidden_dim", [256,384,512,768])
        params["head.mlp.activation"] = trial.suggest_categorical("head.mlp.activation", ["relu","gelu","silu","tanh"])
        params["head.mlp.layernorm"] = trial.suggest_categorical("head.mlp.layernorm", [True, False])
    elif head_type == "glu":
        params["head.glu.hidden_dim"] = trial.suggest_categorical("head.glu.hidden_dim", [256,384,512,768])
        params["head.glu.gate_bias"] = trial.suggest_categorical("head.glu.gate_bias", [True, False])
    elif head_type == "msd":
        params["head.msd.n_samples"] = trial.suggest_int("head.msd.n_samples", 4, 8)
        params["head.msd.alpha"] = trial.suggest_float("head.msd.alpha", 0.1, 0.7)

    loss_name = trial.suggest_categorical("loss.name", ["bce", "weighted_bce", "focal", "asymmetric"])
    if loss_name == "focal":
        params["loss.focal.gamma"] = trial.suggest_float("loss.focal.gamma", 1.0, 4.0)
        params["loss.focal.alpha_pos"] = trial.suggest_float("loss.focal.alpha_pos", 0.1, 0.9)
    elif loss_name == "asymmetric":
        params["loss.asym.gamma_pos"] = trial.suggest_float("loss.asym.gamma_pos", 0.0, 4.0)
        params["loss.asym.gamma_neg"] = trial.suggest_float("loss.asym.gamma_neg", 0.0, 4.0)

    th_policy = trial.suggest_categorical("pred.threshold.policy", ["fixed", "opt_global"])
    if th_policy == "opt_global":
        params["pred.threshold.global"] = trial.suggest_float("pred.threshold.global", 0.2, 0.6)

    params["optim.name"] = trial.suggest_categorical("optim.name", ["adamw"])
    params["optim.lr_encoder"] = trial.suggest_float("optim.lr_encoder", 5e-6, 5e-5, log=True)
    params["optim.lr_head"] = trial.suggest_float("optim.lr_head", 1e-5, 1e-3, log=True)
    params["optim.weight_decay"] = trial.suggest_float("optim.weight_decay", 0.0, 0.2)

    params["sched.name"] = trial.suggest_categorical("sched.name", ["linear", "cosine"])
    params["sched.warmup_ratio"] = trial.suggest_float("sched.warmup_ratio", 0.0, 0.2)
    params["train.grad_clip_norm"] = trial.suggest_float("train.grad_clip_norm", 0.0, 2.0)

    # Optional augmentation knobs (uncomment only if implemented)
    # params["aug.enabled"] = trial.suggest_categorical("aug.enabled", [False, True])
    # if params["aug.enabled"]:
    #     params["aug.policy"] = trial.suggest_categorical("aug.policy", ["nlpaug_x", "textattack_y"])
    #     params["aug.prob"]   = trial.suggest_float("aug.prob", 0.0, 0.5)

    for k in params.keys():
        assert not any(k.startswith(fp) for fp in FORBIDDEN_PARAMS), f"Forbidden HPO param: {k}"
        assert k.startswith(ALLOWED_PARAM_PREFIXES) or k in {"pooling"}, f"Unexpected HPO param: {k}"
    return params

def _clip(lo: float, hi: float, a: float, b: float) -> tuple[float,float]:
    return max(a, lo), min(b, hi)

def narrow_numeric(winner: float, mult_lo: float, mult_hi: float, hard_bounds: tuple[float, float] | None = None) -> tuple[float, float]:
    lo = winner * mult_lo
    hi = winner * mult_hi
    if hard_bounds:
        lo, hi = _clip(hard_bounds[0], hard_bounds[1], lo, hi)
    if hi <= lo:
        midpoint = winner
        if hard_bounds:
            midpoint = max(min(midpoint, hard_bounds[1]), hard_bounds[0])
        eps = max(abs(midpoint) * 1e-3, 1e-6)
        lo = midpoint - eps
        hi = midpoint + eps
        if hard_bounds:
            lo, hi = _clip(hard_bounds[0], hard_bounds[1], lo, hi)
    return lo, hi

def stage_b_space_from_winner(best_params: dict) -> dict:
    space = {}
    # freeze structural keys
    for k in ["head.type","pooling","loss.name","aug.enabled","aug.policy"]:
        if k in best_params:
            space[k] = ("freeze", best_params[k])

    # narrow continuous
    if "optim.lr_encoder" in best_params:
        space["optim.lr_encoder"] = ("float_log",) + narrow_numeric(best_params["optim.lr_encoder"], 0.5, 2.0, (5e-6, 5e-5))
    if "optim.lr_head" in best_params:
        space["optim.lr_head"] = ("float_log",) + narrow_numeric(best_params["optim.lr_head"], 0.5, 2.0, (1e-5, 1e-3))
    if "optim.weight_decay" in best_params:
        space["optim.weight_decay"] = ("float", 1e-6, 1e-2)
    if "head.dropout" in best_params:
        lo, hi = best_params["head.dropout"] - 0.2, best_params["head.dropout"] + 0.2
        lo, hi = max(lo, 0.0), min(hi, 0.5)
        space["head.dropout"] = ("float", lo, hi)
    if "sched.warmup_ratio" in best_params:
        lo, hi = best_params["sched.warmup_ratio"] - 0.05, best_params["sched.warmup_ratio"] + 0.05
        lo, hi = max(lo, 0.0), min(hi, 0.2)
        space["sched.warmup_ratio"] = ("float", lo, hi)
    # losses fine knobs
    if best_params.get("loss.name") == "focal":
        if "loss.focal.gamma" in best_params:
            g = best_params["loss.focal.gamma"]
            space["loss.focal.gamma"] = ("float", max(1.0, g*0.8), min(4.0, g*1.2))
        if "loss.focal.alpha_pos" in best_params:
            a = best_params["loss.focal.alpha_pos"]
            space["loss.focal.alpha_pos"] = ("float", max(0.1, a*0.8), min(0.9, a*1.2))
    if best_params.get("loss.name") == "asymmetric":
        for k in ["loss.asym.gamma_pos","loss.asym.gamma_neg"]:
            if k in best_params:
                v = best_params[k]
                space[k] = ("float", max(0.0, v*0.8), min(4.0, v*1.2))
    # aug
    if best_params.get("aug.enabled", False):
        if "aug.prob" in best_params:
            v = best_params["aug.prob"]
            space["aug.prob"] = ("float", max(0.0, v-0.2), min(0.7, v+0.2))
    return space
