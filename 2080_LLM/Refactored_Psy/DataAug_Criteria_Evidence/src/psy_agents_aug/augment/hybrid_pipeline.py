"""Hybrid augmentation pipeline combining multiple strategies.

This module provides a hybrid augmentation approach that mixes
different augmentation methods with configurable proportions.
"""

import logging
from typing import List, Dict
import random

from .base_augmentor import AugmentationConfig, BaseAugmentor
from .nlpaug_pipeline import NLPAugPipeline, NLPAUG_AVAILABLE
from .textattack_pipeline import TextAttackPipeline, TEXTATTACK_AVAILABLE

logger = logging.getLogger(__name__)


class HybridPipeline(BaseAugmentor):
    """Hybrid augmentation combining multiple methods.
    
    This pipeline allows mixing different augmentation strategies
    with specified proportions (e.g., 50% synonym, 30% insert, 20% swap).
    """
    
    name = "hybrid_pipeline"
    
    def __init__(
        self,
        config: AugmentationConfig,
        mix_proportions: Dict[str, float] = None,
    ):
        """Initialize hybrid pipeline.
        
        Args:
            config: Augmentation configuration
            mix_proportions: Dictionary mapping method names to proportions
                           Example: {"nlpaug_synonym": 0.5, "textattack_wordnet": 0.5}
                           Proportions must sum to 1.0
        """
        super().__init__(config)
        
        # Default mix: 50% nlpaug synonym, 50% textattack wordnet
        if mix_proportions is None:
            mix_proportions = {
                "nlpaug_synonym": 0.5,
                "textattack_wordnet": 0.5,
            }
        
        # Validate proportions
        total = sum(mix_proportions.values())
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Mix proportions must sum to 1.0, got {total}. "
                f"Proportions: {mix_proportions}"
            )
        
        self.mix_proportions = mix_proportions
        self.augmenters = {}
        
        # Initialize augmenters based on mix
        for method_name, proportion in mix_proportions.items():
            if proportion <= 0:
                continue
            
            try:
                augmenter = self._create_augmenter(method_name, config)
                self.augmenters[method_name] = {
                    "augmenter": augmenter,
                    "proportion": proportion,
                }
            except Exception as e:
                logger.warning(f"Failed to initialize {method_name}: {e}")
        
        if not self.augmenters:
            raise ValueError("No augmenters could be initialized")
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        logger.info(
            f"Initialized HybridPipeline with {len(self.augmenters)} augmenters: "
            f"{list(self.augmenters.keys())}"
        )
    
    def _create_augmenter(self, method_name: str, config: AugmentationConfig):
        """Create an augmenter from method name.
        
        Args:
            method_name: Name of augmentation method
            config: Augmentation configuration
            
        Returns:
            Initialized augmenter
        """
        if method_name == "nlpaug_synonym":
            if not NLPAUG_AVAILABLE:
                raise ImportError("nlpaug not available")
            return NLPAugPipeline(config, aug_method="synonym")
        
        elif method_name == "nlpaug_insert":
            if not NLPAUG_AVAILABLE:
                raise ImportError("nlpaug not available")
            return NLPAugPipeline(config, aug_method="insert")
        
        elif method_name == "nlpaug_swap":
            if not NLPAUG_AVAILABLE:
                raise ImportError("nlpaug not available")
            return NLPAugPipeline(config, aug_method="swap")
        
        elif method_name == "textattack_wordnet":
            if not TEXTATTACK_AVAILABLE:
                raise ImportError("TextAttack not available")
            return TextAttackPipeline(config, aug_method="wordnet")
        
        elif method_name == "textattack_embedding":
            if not TEXTATTACK_AVAILABLE:
                raise ImportError("TextAttack not available")
            return TextAttackPipeline(config, aug_method="embedding")
        
        else:
            raise ValueError(f"Unknown augmentation method: {method_name}")
    
    def augment_text(self, text: str, num_variants: int = 1) -> List[str]:
        """Augment a single text using hybrid approach.
        
        Args:
            text: Text to augment
            num_variants: Number of augmented variants to generate
            
        Returns:
            List of augmented texts
        """
        if not text or not text.strip():
            return []
        
        augmented = []
        
        # Distribute variants across augmenters based on proportions
        for method_name, info in self.augmenters.items():
            augmenter = info["augmenter"]
            proportion = info["proportion"]
            
            # Calculate number of variants for this augmenter
            n_variants = max(1, int(num_variants * proportion))
            
            try:
                variants = augmenter.augment_text(text, n_variants)
                augmented.extend(variants)
            except Exception as e:
                logger.warning(f"Failed to augment with {method_name}: {e}")
        
        # Shuffle and limit to requested number
        random.shuffle(augmented)
        return augmented[:num_variants]
