"""Utility helpers for configuring LoRA adapters."""

from __future__ import annotations

from typing import List


def infer_lora_target_modules(backbone) -> List[str]:
    """
    Infer reasonable target module names for LoRA adapters.

    Mirrors the logic used in the classification trainer so both QA and
    classification flows stay consistent when attaching LoRA layers.
    """
    suffixes = {name.split(".")[-1] for name, _ in backbone.named_modules()}
    if {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(suffixes):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if {"gate_proj", "down_proj", "up_proj"}.issubset(suffixes):
        return ["gate_proj", "down_proj", "up_proj"]
    if "c_attn" in suffixes and "c_proj" in suffixes:
        return ["c_attn", "c_proj"]
    if {"W_pack", "o_proj"}.issubset(suffixes):
        return ["W_pack", "o_proj"]

    proj_modules = [name for name in suffixes if "proj" in name]
    return proj_modules or ["c_attn"]
