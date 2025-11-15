from __future__ import annotations

import json
import platform


def _detect_torch_capabilities() -> dict:
    try:
        import torch
    except ModuleNotFoundError:
        return {"installed": False}

    cuda_available = torch.cuda.is_available()
    bf16_supported = False
    if cuda_available and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())  # type: ignore[arg-type]
        except Exception:
            bf16_supported = False

    capability = None
    total_memory = None
    if cuda_available:
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            capability = f"{props.major}.{props.minor}"
            total_memory = props.total_memory // (1024**3)
        except Exception:
            capability = "unknown"

    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "device_capability": capability,
        "total_memory_gb": total_memory,
        "bf16_supported": bf16_supported,
    }


def _detect_optuna() -> dict:
    try:
        import optuna
    except ModuleNotFoundError:
        return {"installed": False}
    return {"installed": True, "version": optuna.__version__}


def main() -> None:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _detect_torch_capabilities(),
        "optuna": _detect_optuna(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
