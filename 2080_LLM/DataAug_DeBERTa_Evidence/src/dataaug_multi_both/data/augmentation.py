"""Text augmentation using TextAttack for evidence augmentation.

This module provides text augmentation methods that preserve non-evidence regions
while augmenting evidence text.

Note: TextAttack is optional. If not installed, augmentation will be disabled.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import TextAttack (optional dependency)
try:
    from textattack.augmentation import (
        WordNetAugmenter,
        EmbeddingAugmenter,
        CharSwapAugmenter,
        EasyDataAugmenter
    )
    TEXTATTACK_AVAILABLE = True
except ImportError:
    TEXTATTACK_AVAILABLE = False
    logger.warning(
        "TextAttack not available. Data augmentation will be disabled. "
        "Install with: pip install textattack"
    )


@dataclass
class AugmentationConfig:
    """Configuration for text augmentation."""
    
    method: str = "synonym"  # synonym, insert, swap, char_perturb, eda
    probability: float = 0.3  # Probability of augmenting each example (0.0-0.5)
    num_augmentations: int = 1  # Number of augmented versions per example
    preserve_non_evidence: bool = True  # Preserve non-evidence regions
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.probability <= 0.5:
            raise ValueError(
                f"Augmentation probability must be in [0.0, 0.5], got {self.probability}"
            )
        
        if self.method not in ["synonym", "insert", "swap", "char_perturb", "eda", "none"]:
            raise ValueError(
                f"Unknown augmentation method: {self.method}. "
                f"Valid methods: synonym, insert, swap, char_perturb, eda, none"
            )


class TextAugmenter:
    """Text augmentation using TextAttack."""
    
    def __init__(self, config: AugmentationConfig):
        """Initialize augmenter.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.augmenter = None
        
        if not TEXTATTACK_AVAILABLE:
            logger.warning("TextAttack not available, augmentation disabled")
            return
        
        if config.method == "none":
            return
        
        # Initialize TextAttack augmenter based on method
        try:
            if config.method == "synonym":
                self.augmenter = WordNetAugmenter(pct_words_to_swap=0.1)
            elif config.method == "insert":
                self.augmenter = EmbeddingAugmenter(
                    transformations_per_example=1
                )
            elif config.method == "swap":
                self.augmenter = EasyDataAugmenter(
                    pct_words_to_swap=0.1,
                    transformations_per_example=1
                )
            elif config.method == "char_perturb":
                self.augmenter = CharSwapAugmenter(
                    pct_words_to_swap=0.1,
                    transformations_per_example=1
                )
            elif config.method == "eda":
                self.augmenter = EasyDataAugmenter(
                    transformations_per_example=1
                )
            
            logger.info(f"Initialized TextAttack augmenter: {config.method}")
        except Exception as e:
            logger.error(f"Failed to initialize augmenter: {e}")
            self.augmenter = None
    
    def augment_text(self, text: str) -> List[str]:
        """Augment a single text.
        
        Args:
            text: Text to augment
            
        Returns:
            List of augmented texts (may be empty if augmentation disabled)
        """
        if self.augmenter is None or not text:
            return []
        
        try:
            augmented = self.augmenter.augment(text)
            # TextAttack returns a list of augmented texts
            if isinstance(augmented, str):
                return [augmented]
            return augmented[:self.config.num_augmentations]
        except Exception as e:
            logger.warning(f"Augmentation failed for text: {e}")
            return []
    
    def augment_evidence(
        self,
        text: str,
        evidence_spans: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[str, Dict[str, any]]:
        """Augment only evidence regions, preserve non-evidence text.
        
        Args:
            text: Full text
            evidence_spans: List of (start, end) character positions for evidence
                           If None, augment entire text
            
        Returns:
            Tuple of (augmented_text, metadata)
            metadata contains:
                - original_evidence: Original evidence text
                - augmented_evidence: Augmented evidence text
                - augmentation_applied: Whether augmentation was applied
        """
        # Decide whether to augment based on probability
        if random.random() > self.config.probability:
            return text, {
                "original_evidence": None,
                "augmented_evidence": None,
                "augmentation_applied": False
            }
        
        if evidence_spans is None or not self.config.preserve_non_evidence:
            # Augment entire text
            augmented_texts = self.augment_text(text)
            if augmented_texts:
                return augmented_texts[0], {
                    "original_evidence": text,
                    "augmented_evidence": augmented_texts[0],
                    "augmentation_applied": True
                }
            return text, {
                "original_evidence": text,
                "augmented_evidence": None,
                "augmentation_applied": False
            }
        
        # Augment only evidence spans
        augmented_text = text
        metadata = {
            "original_evidence": [],
            "augmented_evidence": [],
            "augmentation_applied": False
        }
        
        # Sort spans by start position (reverse order for replacement)
        sorted_spans = sorted(evidence_spans, key=lambda x: x[0], reverse=True)
        
        for start, end in sorted_spans:
            evidence_text = text[start:end]
            augmented_evidence = self.augment_text(evidence_text)
            
            if augmented_evidence:
                # Replace evidence span with augmented version
                augmented_text = (
                    augmented_text[:start] +
                    augmented_evidence[0] +
                    augmented_text[end:]
                )
                metadata["original_evidence"].append(evidence_text)
                metadata["augmented_evidence"].append(augmented_evidence[0])
                metadata["augmentation_applied"] = True
        
        return augmented_text, metadata
    
    def augment_batch(
        self,
        examples: List[Dict[str, any]],
        text_key: str = "text",
        evidence_key: Optional[str] = "evidence_spans"
    ) -> List[Dict[str, any]]:
        """Augment a batch of examples.
        
        Args:
            examples: List of example dictionaries
            text_key: Key for text field in examples
            evidence_key: Key for evidence spans field (optional)
            
        Returns:
            List of augmented examples (original + augmented)
        """
        augmented_examples = []
        
        for example in examples:
            # Keep original
            augmented_examples.append(example)
            
            # Augment if probability allows
            if random.random() <= self.config.probability:
                text = example.get(text_key, "")
                evidence_spans = example.get(evidence_key) if evidence_key else None
                
                augmented_text, metadata = self.augment_evidence(text, evidence_spans)
                
                if metadata["augmentation_applied"]:
                    # Create augmented example
                    augmented_example = example.copy()
                    augmented_example[text_key] = augmented_text
                    augmented_example["augmentation_metadata"] = metadata
                    augmented_examples.append(augmented_example)
        
        return augmented_examples


def create_augmenter(
    method: str = "synonym",
    probability: float = 0.3,
    **kwargs
) -> TextAugmenter:
    """Create a text augmenter with given configuration.
    
    Args:
        method: Augmentation method
        probability: Augmentation probability
        **kwargs: Additional configuration options
        
    Returns:
        Configured TextAugmenter
        
    Example:
        augmenter = create_augmenter(method="synonym", probability=0.3)
        augmented_text, metadata = augmenter.augment_evidence(text)
    """
    config = AugmentationConfig(
        method=method,
        probability=probability,
        **kwargs
    )
    return TextAugmenter(config)

