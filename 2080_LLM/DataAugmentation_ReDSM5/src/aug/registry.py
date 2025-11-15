"""
Registry for 28 augmentation methods with support for both nlpaug and textattack.

This module provides a central registry for all augmenters, loading configurations
from YAML and instantiating augmenters with appropriate parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path
import random
import string
import unicodedata
import re

import yaml
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas


class _BaseSimpleAugmenter:
    """Minimal augmenter interface compatible with pipeline expectations."""

    def augment(self, data: Any, n: int = 1) -> List[str]:
        if isinstance(data, (list, tuple)):
            return [self._augment_once(item) for item in data]
        return [self._augment_once(data) for _ in range(max(1, n))]

    def _augment_once(self, text: str) -> str:  # pragma: no cover - abstract hook
        raise NotImplementedError


class WhitespaceJitterAug(_BaseSimpleAugmenter):
    """Randomly removes or duplicates whitespace groups."""

    def __init__(self, jitter_prob: float = 0.1, max_extra_spaces: int = 1) -> None:
        self.jitter_prob = max(0.0, min(1.0, jitter_prob))
        self.max_extra_spaces = max(1, int(max_extra_spaces))

    def _augment_once(self, text: str) -> str:
        tokens = re.split(r"(\s+)", text)
        if not tokens:
            return text

        augmented: List[str] = []
        for token in tokens:
            if not token:
                continue
            if token.isspace():
                if random.random() < self.jitter_prob:
                    if random.random() < 0.5:
                        # Drop this whitespace span entirely.
                        continue
                    # Duplicate whitespace span by adding extra spaces.
                    extra = " " * random.randint(1, self.max_extra_spaces)
                    augmented.append(token + extra)
                    continue
            augmented.append(token)

        result = "".join(augmented)
        return result if result else text


class CasingJitterAug(_BaseSimpleAugmenter):
    """Randomly flips character casing."""

    def __init__(self, aug_char_p: float = 0.1) -> None:
        self.aug_char_p = max(0.0, min(1.0, aug_char_p))

    def _augment_once(self, text: str) -> str:
        chars: List[str] = []
        for ch in text:
            if ch.isalpha() and random.random() < self.aug_char_p:
                chars.append(ch.swapcase())
            else:
                chars.append(ch)
        return "".join(chars)


class RemovePunctuationAug(_BaseSimpleAugmenter):
    """Removes punctuation characters while optionally keeping sentence endings."""

    def __init__(
        self,
        keep_sentence_end: bool = True,
        punctuation: Optional[str] = None,
    ) -> None:
        self.keep_sentence_end = keep_sentence_end
        self.punctuation = punctuation or string.punctuation
        self._sentence_end = {".", "!", "?"}

    def _should_keep(self, ch: str) -> bool:
        if ch not in self.punctuation:
            return True
        if self.keep_sentence_end and ch in self._sentence_end:
            return True
        return False

    def _augment_once(self, text: str) -> str:
        return "".join(ch for ch in text if self._should_keep(ch))


class PunctuationNoiseAug(_BaseSimpleAugmenter):
    """Adds or removes punctuation characters with a specified probability."""

    def __init__(
        self,
        aug_p: float = 0.1,
        punctuation: Optional[Sequence[str]] = None,
    ) -> None:
        self.aug_p = max(0.0, min(1.0, aug_p))
        if punctuation is None:
            punctuation = string.punctuation
        self.punctuation = list(punctuation)

    def _augment_once(self, text: str) -> str:
        result: List[str] = []
        for ch in text:
            if ch in self.punctuation:
                # Randomly drop existing punctuation.
                if random.random() < self.aug_p:
                    # Skip this punctuation character.
                    if random.random() < 0.5:
                        continue
                    # Duplicate punctuation by inserting another random symbol.
                    result.append(random.choice(self.punctuation))
                result.append(ch)
            else:
                result.append(ch)
                # Randomly inject punctuation after non-punctuation characters.
                if random.random() < self.aug_p:
                    result.append(random.choice(self.punctuation))
        return "".join(result) or text


class UnicodeNormalizeAug(_BaseSimpleAugmenter):
    """Normalises unicode strings using Python's unicodedata helpers."""

    def __init__(self, form: str = "NFKC") -> None:
        self.form = form

    def _augment_once(self, text: str) -> str:
        return unicodedata.normalize(self.form, text)


class TextAttackAugmenterAdapter:
    """Wraps a TextAttack augmenter to provide a uniform interface."""

    def __init__(self, augmenter: Any) -> None:
        self.augmenter = augmenter

    def augment(self, data: Any, n: int = 1) -> List[str]:
        if isinstance(data, (list, tuple)):
            return [
                self._augment_single(str(item), max(1, n))[0]
                for item in data
            ]
        return self._augment_single(str(data), n)

    def _augment_single(self, text: str, n: int) -> List[str]:
        collected: List[str] = []
        max_attempts = max(3, n * 3)
        attempts = 0

        while len(collected) < n and attempts < max_attempts:
            generated = self.augmenter.augment(text)
            attempts += 1

            if not generated:
                continue

            if isinstance(generated, tuple):
                generated = generated[0]

            if isinstance(generated, list):
                candidates = generated
            elif isinstance(generated, str):
                candidates = [generated]
            else:  # pragma: no cover - defensive branch
                raise TypeError(
                    f"Unexpected return type from TextAttack augmenter: {type(generated)}"
                )

            for candidate in candidates:
                candidate = candidate.strip()
                if not candidate:
                    continue
                if candidate not in collected:
                    collected.append(candidate)
                    if len(collected) >= n:
                        break

        return collected or [text]


class AugmenterRegistry:
    """
    Registry for all 28 augmentation methods.
    
    Loads augmenter configurations from YAML and provides methods to:
    - List available augmenters
    - Get augmenter instances
    - Query augmenter metadata (stage, library, parameters)
    
    Attributes:
        config_path: Path to augmenters_28.yaml
        augmenters: Dictionary of augmenter configurations
    """
    
    def __init__(self, config_path: str = "configs/augmenters_28.yaml"):
        """
        Initialize registry from YAML config.
        
        Args:
            config_path: Path to augmenters configuration file
        """
        self.config_path = Path(config_path)
        self.augmenters = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Dict[str, Any]]:
        """Load augmenter configurations from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        # Convert list to dictionary keyed by name
        augmenters = {aug["name"]: aug for aug in config["augmenters"]}
        
        return augmenters
    
    def _validate_config(self) -> None:
        """Validate that config contains exactly 28 augmenters."""
        if len(self.augmenters) != 28:
            raise ValueError(
                f"Expected 28 augmenters, found {len(self.augmenters)}"
            )
        
        # Validate required fields
        required_fields = ["name", "lib", "stage", "defaults"]
        for name, config in self.augmenters.items():
            for field in required_fields:
                if field not in config:
                    raise ValueError(
                        f"Augmenter '{name}' missing required field: {field}"
                    )
    
    def list_augmenters(
        self,
        stage: Optional[str] = None,
        lib: Optional[str] = None,
    ) -> List[str]:
        """
        List available augmenters, optionally filtered by stage or library.
        
        Args:
            stage: Filter by stage (char, word, contextual, backtranslation, format)
            lib: Filter by library (nlpaug, textattack)
            
        Returns:
            List of augmenter names
        """
        augmenters = []
        
        for name, config in self.augmenters.items():
            if stage and config["stage"] != stage:
                continue
            if lib and config["lib"] != lib:
                continue
            augmenters.append(name)
        
        return sorted(augmenters)
    
    def get_augmenter_config(self, name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific augmenter.
        
        Args:
            name: Augmenter name
            
        Returns:
            Augmenter configuration dictionary
        """
        if name not in self.augmenters:
            raise ValueError(f"Unknown augmenter: {name}")
        
        return self.augmenters[name].copy()
    
    def get_augmenter_stage(self, name: str) -> str:
        """Get the stage of an augmenter."""
        return self.get_augmenter_config(name)["stage"]
    
    def get_augmenter_lib(self, name: str) -> str:
        """Get the library of an augmenter."""
        return self.get_augmenter_config(name)["lib"]
    
    def get_default_params(self, name: str) -> Dict[str, Any]:
        """Get default parameters for an augmenter."""
        return self.get_augmenter_config(name)["defaults"].copy()
    
    def get_param_space(self, name: str) -> Dict[str, Any]:
        """Get tunable parameter space for an augmenter."""
        config = self.get_augmenter_config(name)
        return config.get("param_space", {}).copy()
    
    def instantiate_augmenter(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """
        Instantiate an augmenter with given parameters.
        
        Args:
            name: Augmenter name
            params: Parameters (uses defaults if None)
            seed: Random seed for reproducibility
            
        Returns:
            Augmenter instance (nlpaug or textattack)
        """
        config = self.get_augmenter_config(name)
        lib = config["lib"]
        stage = config["stage"]
        
        # Merge params with defaults
        full_params = {**config["defaults"]}
        if params:
            full_params.update(params)
        
        # Add seed if supported
        if seed is not None:
            full_params["seed"] = seed
        
        # Instantiate based on library and augmenter type
        if lib == "nlpaug":
            return self._instantiate_nlpaug(name, stage, full_params)
        elif lib == "textattack":
            return self._instantiate_textattack(name, stage, full_params)
        else:
            raise ValueError(f"Unknown library: {lib}")
    
    def _instantiate_nlpaug(
        self,
        name: str,
        stage: str,
        params: Dict[str, Any],
    ) -> Any:
        """Instantiate nlpaug augmenter."""
        params = params.copy()
        # Character-level augmenters
        if name == "random_delete":
            return nac.RandomCharAug(action="delete", **params)
        elif name == "random_insert":
            return nac.RandomCharAug(action="insert", **params)
        elif name == "random_swap":
            return nac.RandomCharAug(action="swap", **params)
        elif name == "keyboard_error":
            return nac.KeyboardAug(**params)
        elif name == "ocr_noise":
            return nac.OcrAug(**params)
        elif name == "char_substitute":
            return nac.RandomCharAug(action="substitute", **params)
        
        # Word-level augmenters
        elif name == "word_dropout":
            return naw.RandomWordAug(action="delete", **params)
        elif name == "word_swap":
            return naw.RandomWordAug(action="swap", **params)
        elif name == "wordnet_synonym":
            return naw.SynonymAug(aug_src="wordnet", **params)
        elif name == "embedding_substitute":
            return naw.WordEmbsAug(**params)
        elif name == "spelling_noise":
            return naw.SpellingAug(**params)
        
        # Contextual augmenters
        elif name == "mlm_infill_bert":
            return naw.ContextualWordEmbsAug(
                model_path=params.pop("model_path"),
                action="substitute",
                **params
            )
        elif name == "mlm_infill_roberta":
            return naw.ContextualWordEmbsAug(
                model_path=params.pop("model_path"),
                action="substitute",
                **params
            )
        elif name == "contextual_substitute":
            return naw.ContextualWordEmbsAug(
                model_path=params.pop("model_path"),
                action="substitute",
                **params
            )
        elif name == "paraphrase_t5":
            return nas.AbstSummAug(
                model_path=params.pop("model_path"),
                **params
            )
        
        # Back-translation augmenters
        elif name.startswith("en_") and name.endswith("_en"):
            return nas.BackTranslationAug(
                from_model_name=params.pop("from_model"),
                to_model_name=params.pop("to_model"),
                **params
            )
        
        # Formatting augmenters
        elif name == "punctuation_noise":
            params.pop("seed", None)
            return PunctuationNoiseAug(**params)
        elif name == "contraction_expand":
            return naw.ContractionAug(aug_direction="expand", **params)
        elif name == "contraction_collapse":
            return naw.ContractionAug(aug_direction="collapse", **params)
        
        else:
            raise NotImplementedError(f"Augmenter '{name}' not yet implemented")
    
    def _instantiate_textattack(
        self,
        name: str,
        stage: str,
        params: Dict[str, Any],
    ) -> Any:
        """Instantiate textattack augmenter."""
        params = params.copy()
        try:
            from textattack.augmentation import CharSwapAugmenter  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "textattack is required for textattack-based augmenters. "
                "Install it via `pip install textattack`."
            ) from exc
        params.pop("seed", None)

        if name == "whitespace_jitter":
            return WhitespaceJitterAug(**params)
        elif name == "casing_jitter":
            return CasingJitterAug(**params)
        elif name == "remove_punctuation":
            return RemovePunctuationAug(**params)
        elif name == "normalize_unicode":
            return UnicodeNormalizeAug(**params)
        elif name == "add_typos":
            aug_p = params.pop("aug_p", None)
            if aug_p is not None and "pct_words_to_swap" not in params:
                params["pct_words_to_swap"] = max(0.0, min(1.0, aug_p))
            augmenter = CharSwapAugmenter(**params)
            return TextAttackAugmenterAdapter(augmenter)
        else:
            raise NotImplementedError(f"TextAttack augmenter '{name}' not yet implemented")
    
    def get_stage_distribution(self) -> Dict[str, int]:
        """Get count of augmenters per stage."""
        distribution = {}
        for config in self.augmenters.values():
            stage = config["stage"]
            distribution[stage] = distribution.get(stage, 0) + 1
        return distribution
    
    def get_library_distribution(self) -> Dict[str, int]:
        """Get count of augmenters per library."""
        distribution = {}
        for config in self.augmenters.values():
            lib = config["lib"]
            distribution[lib] = distribution.get(lib, 0) + 1
        return distribution
