"""NLPAug-based text augmentation pipeline.

This module provides augmentation using the nlpaug library for:
- Synonym replacement (WordNet)
- Random word insertion
- Random word swap
"""

import logging
from typing import List
import random

from .base_augmentor import AugmentationConfig, BaseAugmentor

logger = logging.getLogger(__name__)

# Try to import nlpaug (optional dependency)
try:
    import nlpaug.augmenter.word as naw
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False
    logger.warning(
        "nlpaug not available. Install with: pip install nlpaug"
    )


class NLPAugPipeline(BaseAugmentor):
    """Text augmentation using nlpaug library.
    
    Supports multiple augmentation strategies:
    - synonym: Replace words with synonyms from WordNet
    - insert: Insert random words
    - swap: Swap word positions
    """
    
    name = "nlpaug_pipeline"
    
    def __init__(
        self,
        config: AugmentationConfig,
        aug_method: str = "synonym",
        aug_min: int = 1,
        aug_max: int = 3,
    ):
        """Initialize NLPAug pipeline.
        
        Args:
            config: Augmentation configuration
            aug_method: Augmentation method ("synonym", "insert", "swap")
            aug_min: Minimum number of words to augment
            aug_max: Maximum number of words to augment
        """
        super().__init__(config)
        
        if not NLPAUG_AVAILABLE:
            raise ImportError(
                "nlpaug is required for NLPAugPipeline. "
                "Install it via `pip install nlpaug`"
            )
        
        self.aug_method = aug_method
        self.aug_min = aug_min
        self.aug_max = aug_max
        
        # Initialize augmenter based on method
        if aug_method == "synonym":
            self.augmenter = naw.SynonymAug(
                aug_src="wordnet",
                aug_min=aug_min,
                aug_max=aug_max
            )
        elif aug_method == "insert":
            self.augmenter = naw.ContextualWordEmbsAug(
                model_path="bert-base-uncased",
                action="insert",
                aug_min=aug_min,
                aug_max=aug_max
            )
        elif aug_method == "swap":
            self.augmenter = naw.RandomWordAug(
                action="swap",
                aug_min=aug_min,
                aug_max=aug_max
            )
        else:
            raise ValueError(
                f"Unknown augmentation method: {aug_method}. "
                f"Valid methods: synonym, insert, swap"
            )
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        logger.info(
            f"Initialized NLPAugPipeline with method={aug_method}, "
            f"aug_range=[{aug_min}, {aug_max}]"
        )
    
    def augment_text(self, text: str, num_variants: int = 1) -> List[str]:
        """Augment a single text using nlpaug.
        
        Args:
            text: Text to augment
            num_variants: Number of augmented variants to generate
            
        Returns:
            List of augmented texts
        """
        if not text or not text.strip():
            return []
        
        augmented = []
        seen = set()
        attempts = 0
        max_attempts = num_variants * 4
        
        while len(augmented) < num_variants and attempts < max_attempts:
            attempts += 1
            
            try:
                result = self.augmenter.augment(text)
                
                # Handle different return types
                if isinstance(result, str):
                    candidate = result
                elif isinstance(result, list) and len(result) > 0:
                    candidate = result[0]
                else:
                    continue
                
                candidate = candidate.strip()
                
                # Skip if empty or identical to original
                if not candidate or candidate.lower() == text.lower():
                    continue
                
                # Skip if already seen
                if candidate in seen:
                    continue
                
                seen.add(candidate)
                augmented.append(candidate)
                
            except Exception as e:
                logger.debug(f"Augmentation attempt failed: {e}")
                continue
        
        return augmented
