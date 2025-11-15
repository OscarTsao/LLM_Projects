"""Back-translation augmentation pipeline (optional).

This module provides back-translation augmentation for text data.
Translates text to an intermediate language and back to English.

Note: Requires additional dependencies (transformers with translation models).
"""

import logging
from typing import List
import random

from .base_augmentor import AugmentationConfig, BaseAugmentor

logger = logging.getLogger(__name__)

# Try to import translation models (optional)
try:
    from transformers import MarianMTModel, MarianTokenizer
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    logger.warning(
        "Translation models not available. Back-translation will be disabled."
    )


class BackTranslationPipeline(BaseAugmentor):
    """Back-translation augmentation pipeline.
    
    Translates text to an intermediate language (e.g., German, French)
    and back to English to generate paraphrases.
    """
    
    name = "backtranslation_pipeline"
    
    def __init__(
        self,
        config: AugmentationConfig,
        intermediate_lang: str = "de",  # German by default
    ):
        """Initialize back-translation pipeline.
        
        Args:
            config: Augmentation configuration
            intermediate_lang: Intermediate language code ("de" for German, "fr" for French)
        """
        super().__init__(config)
        
        if not TRANSLATION_AVAILABLE:
            raise ImportError(
                "Translation models not available. "
                "Install transformers with translation support."
            )
        
        self.intermediate_lang = intermediate_lang
        
        # Initialize translation models
        try:
            # English to intermediate language
            model_name_forward = f"Helsinki-NLP/opus-mt-en-{intermediate_lang}"
            self.tokenizer_forward = MarianTokenizer.from_pretrained(model_name_forward)
            self.model_forward = MarianMTModel.from_pretrained(model_name_forward)
            
            # Intermediate language to English
            model_name_backward = f"Helsinki-NLP/opus-mt-{intermediate_lang}-en"
            self.tokenizer_backward = MarianTokenizer.from_pretrained(model_name_backward)
            self.model_backward = MarianMTModel.from_pretrained(model_name_backward)
            
            logger.info(f"Initialized BackTranslation with intermediate_lang={intermediate_lang}")
        except Exception as e:
            raise RuntimeError(f"Failed to load translation models: {e}")
        
        # Set random seed for reproducibility
        random.seed(config.seed)
    
    def _translate(self, text: str, tokenizer, model) -> str:
        """Translate text using given model.
        
        Args:
            text: Text to translate
            tokenizer: Tokenizer for the model
            model: Translation model
            
        Returns:
            Translated text
        """
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
    
    def augment_text(self, text: str, num_variants: int = 1) -> List[str]:
        """Augment a single text using back-translation.
        
        Args:
            text: Text to augment
            num_variants: Number of augmented variants to generate
            
        Returns:
            List of augmented texts
        """
        if not text or not text.strip():
            return []
        
        augmented = []
        
        try:
            # Translate to intermediate language
            intermediate = self._translate(text, self.tokenizer_forward, self.model_forward)
            
            # Translate back to English
            back_translated = self._translate(intermediate, self.tokenizer_backward, self.model_backward)
            
            back_translated = back_translated.strip()
            
            # Only add if different from original
            if back_translated and back_translated.lower() != text.lower():
                augmented.append(back_translated)
        
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
        
        # For multiple variants, try with different random seeds
        # (results may vary slightly due to beam search randomness)
        return augmented[:num_variants]
