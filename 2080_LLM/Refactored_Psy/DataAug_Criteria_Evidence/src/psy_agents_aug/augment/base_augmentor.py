"""Base augmentation interface for text augmentation pipelines.

This module provides a unified interface for all augmentation strategies,
ensuring consistent behavior across different augmentation methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for text augmentation.
    
    Attributes:
        enabled: Whether augmentation is enabled
        pipeline: Name of the augmentation pipeline to use
        ratio: Proportion of training data to augment (0.0-1.0)
        max_aug_per_sample: Maximum augmented variants per sample
        seed: Random seed for reproducibility
        preserve_balance: Whether to preserve class balance
        train_only: Only apply augmentation to training split (CRITICAL)
    """
    enabled: bool = False
    pipeline: str = "nlpaug_pipeline"
    ratio: float = 0.5
    max_aug_per_sample: int = 1
    seed: int = 42
    preserve_balance: bool = True
    train_only: bool = True  # CRITICAL: Never augment val/test
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"Augmentation ratio must be in [0.0, 1.0], got {self.ratio}")
        
        if self.max_aug_per_sample < 1:
            raise ValueError(f"max_aug_per_sample must be >= 1, got {self.max_aug_per_sample}")
        
        if not self.train_only:
            logger.warning(
                "CRITICAL: train_only is False! Augmentation should ONLY apply to training data. "
                "Setting train_only=True to prevent data leakage."
            )
            self.train_only = True


class BaseAugmentor(ABC):
    """Base class for all text augmentation strategies.
    
    All augmentors must:
    1. Only augment training data (NEVER val/test)
    2. Maintain deterministic behavior (same seed = same augmentations)
    3. Preserve class balance when configured
    4. Handle edge cases gracefully (empty text, special characters, etc.)
    """
    
    name: str = "base"
    
    def __init__(self, config: AugmentationConfig):
        """Initialize augmentor.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self._verify_train_only_guarantee()
    
    def _verify_train_only_guarantee(self):
        """Verify that augmentation will only apply to training data."""
        if not self.config.train_only:
            raise ValueError(
                "CRITICAL: Augmentation MUST only apply to training data. "
                "Set config.train_only=True"
            )
    
    @abstractmethod
    def augment_text(self, text: str, num_variants: int = 1) -> List[str]:
        """Augment a single text.
        
        Args:
            text: Text to augment
            num_variants: Number of augmented variants to generate
            
        Returns:
            List of augmented texts (may be fewer than num_variants)
        """
        raise NotImplementedError
    
    def augment_batch(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        split: str = "train"
    ) -> tuple[List[str], Optional[List[int]]]:
        """Augment a batch of texts.
        
        CRITICAL: Only augments if split == "train"
        
        Args:
            texts: List of texts to augment
            labels: Optional list of labels (for balance preservation)
            split: Data split name ("train", "val", "test")
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
            If split != "train", returns original (texts, labels)
        """
        # CRITICAL: Only augment training data
        if split != "train":
            logger.info(f"Skipping augmentation for split '{split}' (train_only=True)")
            return texts, labels
        
        if not self.config.enabled:
            logger.info("Augmentation disabled")
            return texts, labels
        
        augmented_texts = []
        augmented_labels = [] if labels is not None else None
        
        for idx, text in enumerate(texts):
            # Keep original
            augmented_texts.append(text)
            if labels is not None:
                augmented_labels.append(labels[idx])
            
            # Generate augmented variants
            try:
                variants = self.augment_text(text, self.config.max_aug_per_sample)
                augmented_texts.extend(variants)
                if labels is not None:
                    augmented_labels.extend([labels[idx]] * len(variants))
            except Exception as e:
                logger.warning(f"Failed to augment text at index {idx}: {e}")
        
        return augmented_texts, augmented_labels
    
    def get_augmentation_stats(self) -> dict:
        """Get statistics about augmentation configuration.
        
        Returns:
            Dictionary with augmentation stats
        """
        return {
            "augmentor": self.name,
            "enabled": self.config.enabled,
            "ratio": self.config.ratio,
            "max_aug_per_sample": self.config.max_aug_per_sample,
            "train_only": self.config.train_only,
            "preserve_balance": self.config.preserve_balance,
        }
