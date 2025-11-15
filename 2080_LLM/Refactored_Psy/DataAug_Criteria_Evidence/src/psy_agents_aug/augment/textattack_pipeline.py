"""TextAttack-based text augmentation pipeline.

This module provides augmentation using the TextAttack library for:
- WordNet synonym replacement
- Embedding-based word replacement
"""

import logging
from typing import List
import random

from .base_augmentor import AugmentationConfig, BaseAugmentor

logger = logging.getLogger(__name__)

# Try to import TextAttack (optional dependency)
try:
    from textattack.augmentation import (
        WordNetAugmenter,
        EmbeddingAugmenter,
    )
    TEXTATTACK_AVAILABLE = True
except ImportError:
    TEXTATTACK_AVAILABLE = False
    logger.warning(
        "TextAttack not available. Install with: pip install textattack"
    )


class TextAttackPipeline(BaseAugmentor):
    """Text augmentation using TextAttack library.
    
    Supports multiple augmentation strategies:
    - wordnet: Replace words with WordNet synonyms
    - embedding: Replace words using word embeddings
    """
    
    name = "textattack_pipeline"
    
    def __init__(
        self,
        config: AugmentationConfig,
        aug_method: str = "wordnet",
        pct_words_to_swap: float = 0.1,
    ):
        """Initialize TextAttack pipeline.
        
        Args:
            config: Augmentation configuration
            aug_method: Augmentation method ("wordnet", "embedding")
            pct_words_to_swap: Percentage of words to swap (0.0-1.0)
        """
        super().__init__(config)
        
        if not TEXTATTACK_AVAILABLE:
            raise ImportError(
                "TextAttack is required for TextAttackPipeline. "
                "Install it via `pip install textattack`"
            )
        
        self.aug_method = aug_method
        self.pct_words_to_swap = pct_words_to_swap
        
        # Initialize augmenter based on method
        if aug_method == "wordnet":
            self.augmenter = WordNetAugmenter(
                pct_words_to_swap=pct_words_to_swap
            )
        elif aug_method == "embedding":
            self.augmenter = EmbeddingAugmenter(
                transformations_per_example=1
            )
        else:
            raise ValueError(
                f"Unknown augmentation method: {aug_method}. "
                f"Valid methods: wordnet, embedding"
            )
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        logger.info(
            f"Initialized TextAttackPipeline with method={aug_method}, "
            f"pct_words_to_swap={pct_words_to_swap}"
        )
    
    def augment_text(self, text: str, num_variants: int = 1) -> List[str]:
        """Augment a single text using TextAttack.
        
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
