"""
Method registry and augmentation wrappers for the augmentation-only pipeline.

This module loads augmentation method specifications from YAML and produces
augmenter instances built on top of nlpaug and textattack. It also tracks
instantiation failures so the CLI can surface unavailable methods up-front.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import threading

import yaml

@dataclass(frozen=True)
class MethodSpec:
    """Specification describing a single augmentation method."""

    id: str
    lib: str
    kind: str
    args: Dict[str, Any]
    requires_gpu: bool = False

    def __post_init__(self) -> None:
        normalized_args = dict(self.args)
        object.__setattr__(self, "args", normalized_args)


class MethodUnavailableError(RuntimeError):
    """Raised when an augmentation method cannot be instantiated."""

    def __init__(self, method_id: str, reason: str) -> None:
        super().__init__(f"{method_id}: {reason}")
        self.method_id = method_id
        self.reason = reason


def load_method_specs(path: str | Path) -> List[MethodSpec]:
    """
    Load method specifications from YAML.

    The YAML is expected to contain a top-level ``methods`` list where each
    element provides ``id``, ``lib``, ``kind``, ``args``, and ``requires_gpu``.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Augmentation registry not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    methods_section = raw.get("methods")
    if not isinstance(methods_section, list):
        raise ValueError("Configuration must define a top-level 'methods' list.")

    specs: List[MethodSpec] = []
    for entry in methods_section:
        if not isinstance(entry, dict):
            raise ValueError("Each method entry must be a mapping.")
        try:
            spec = MethodSpec(
                id=str(entry["id"]),
                lib=str(entry["lib"]),
                kind=str(entry["kind"]),
                args=entry.get("args", {}) or {},
                requires_gpu=bool(entry.get("requires_gpu", False)),
            )
        except KeyError as exc:
            raise ValueError(f"Missing required field {exc} in method spec.") from exc
        specs.append(spec)
    return specs


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_device(spec: MethodSpec, override: Optional[str] = None) -> str:
    if override:
        return override
    if spec.requires_gpu and _cuda_available():
        return "cuda"
    args_device = spec.args.get("device")
    if isinstance(args_device, str) and args_device.lower() == "auto":
        return "cuda" if _cuda_available() else "cpu"
    return "cpu"


class _LazyAugmenter:
    """Defers heavy augmenter construction until first use."""

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._instance: Any | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> Any:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def augment(self, text: Any, n: int = 1) -> List[str]:
        instance = self._ensure()
        augment = getattr(instance, "augment", None)
        if augment is None:  # pragma: no cover - defensive
            raise AttributeError("Augmenter must provide an 'augment' method.")
        return augment(text, n=n)


class _TextAttackWrapper:
    """Uniform augmentation interface for TextAttack transformations."""

    def __init__(self, transformation: Any, **augmenter_kwargs: Any) -> None:
        from textattack.augmentation import Augmenter  # type: ignore

        self.transformation = transformation
        self._augmenter = Augmenter(
            transformation=transformation,
            **augmenter_kwargs,
        )

    def augment(self, text: str, n: int = 1) -> List[str]:
        results: List[str] = []
        attempts = 0
        while len(results) < n and attempts < max(5, n * 3):
            attempts += 1
            generated = self._augmenter.augment(text)
            if not generated:
                continue
            if isinstance(generated, str):
                candidates = [generated]
            else:
                candidates = list(generated)
            for candidate in candidates:
                if not isinstance(candidate, str):
                    continue
                candidate = candidate.strip()
                if not candidate or candidate.lower() == text.lower():
                    continue
                results.append(candidate)
                if len(results) >= n:
                    break
        if not results:
            return [text]
        return results[:n]


class MethodRegistry:
    """Central registry responsible for instantiating configured augmenters."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self._specs = {spec.id: spec for spec in load_method_specs(config_path)}
        self._instances: MutableMapping[str, Any] = {}
        self._failures: MutableMapping[str, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public
    @property
    def specs(self) -> Mapping[str, MethodSpec]:
        return self._specs

    def list_methods(self) -> List[str]:
        """Return method ids sorted deterministically."""
        return sorted(self._specs.keys())

    def missing_methods(self) -> Mapping[str, str]:
        """Return methods that failed to instantiate and their reasons."""
        return dict(self._failures)

    def is_available(self, method_id: str) -> bool:
        return method_id in self._specs and method_id not in self._failures

    def get_spec(self, method_id: str) -> MethodSpec:
        try:
            return self._specs[method_id]
        except KeyError as exc:
            raise KeyError(f"Unknown method id '{method_id}'.") from exc

    def instantiate(self, method_id: str) -> Any:
        """
        Instantiate (or reuse) an augmenter for the requested method.

        Returns the augmenter instance if available. Raises MethodUnavailableError
        if the method cannot be constructed for the current environment.
        """
        if method_id in self._failures:
            raise MethodUnavailableError(method_id, self._failures[method_id])

        if method_id not in self._instances:
            spec = self.get_spec(method_id)
            try:
                instance = self._build_instance(spec)
            except MethodUnavailableError as exc:
                self._failures[method_id] = exc.reason
                raise
            except Exception as exc:  # pragma: no cover - defensive
                reason = f"unexpected error during instantiation: {exc}"
                self._failures[method_id] = reason
                raise MethodUnavailableError(method_id, reason) from exc
            with self._lock:
                self._instances[method_id] = instance
        return self._instances[method_id]

    # ----------------------------------------------------------------- private
    def _build_instance(self, spec: MethodSpec) -> Any:
        if spec.lib == "nlpaug":
            return self._instantiate_nlpaug(spec)
        if spec.lib == "textattack":
            return self._instantiate_textattack(spec)
        raise MethodUnavailableError(spec.id, f"Unsupported library: {spec.lib}")

    def _instantiate_nlpaug(self, spec: MethodSpec) -> Any:
        module_path, class_name = _split_kind(spec.kind)

        try:
            import_module(module_path)
        except ImportError as exc:  # pragma: no cover - import guard
            raise MethodUnavailableError(
                spec.id, f"Cannot import module '{module_path}': {exc}"
            ) from exc

        def factory() -> Any:
            try:
                module = import_module(module_path)
            except ImportError as exc:  # pragma: no cover - import guard
                raise MethodUnavailableError(
                    spec.id, f"Cannot import module '{module_path}': {exc}"
                ) from exc
            try:
                cls = getattr(module, class_name)
            except AttributeError as exc:  # pragma: no cover - defensive
                raise MethodUnavailableError(
                    spec.id, f"Module '{module_path}' has no attribute '{class_name}'."
                ) from exc

            args = dict(spec.args)
            if "device" in args and isinstance(args["device"], str):
                if args["device"].lower() == "auto":
                    args["device"] = _resolve_device(spec)
            elif spec.requires_gpu:
                args.setdefault("device", _resolve_device(spec))
            return cls(**args)

        return _LazyAugmenter(factory)

    def _instantiate_textattack(self, spec: MethodSpec) -> Any:
        module_path, class_name = _split_kind(spec.kind)
        args = dict(spec.args)
        augmentation_args = args.pop("augmenter_kwargs", {})

        try:
            import_module(module_path)
        except ImportError as exc:
            raise MethodUnavailableError(
                spec.id, f"Cannot import module '{module_path}': {exc}"
            ) from exc

        def factory() -> Any:
            try:
                module = import_module(module_path)
            except ImportError as exc:
                raise MethodUnavailableError(
                    spec.id, f"Cannot import module '{module_path}': {exc}"
                ) from exc
            try:
                cls = getattr(module, class_name)
            except AttributeError as exc:
                raise MethodUnavailableError(
                    spec.id,
                    f"Module '{module_path}' has no attribute '{class_name}'.",
                ) from exc
            transformation = cls(**args)
            return _TextAttackWrapper(transformation, **augmentation_args)

        return _LazyAugmenter(factory)


@lru_cache(maxsize=64)
def _split_kind(kind: str) -> tuple[str, str]:
    if "." not in kind:
        raise ValueError(f"Augmenter kind must be fully qualified. Got '{kind}'.")
    module_path, class_name = kind.rsplit(".", 1)
    return module_path, class_name


def list_missing_methods(registry: MethodRegistry) -> Dict[str, str]:
    """
    Attempt to instantiate each configured method and collect failures.

    Returns a dictionary mapping missing method ids to a human-readable reason.
    """
    missing: Dict[str, str] = {}
    for method_id in registry.list_methods():
        if registry.is_available(method_id) and method_id not in missing:
            try:
                registry.instantiate(method_id)
            except MethodUnavailableError as exc:
                missing[method_id] = exc.reason
    missing.update(registry.missing_methods())
    return missing
