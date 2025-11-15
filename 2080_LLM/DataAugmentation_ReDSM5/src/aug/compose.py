"""
Augmentation pipeline composition for chaining multiple augmenters.

Supports:
- Sequential application of augmenters
- Deterministic seeding for reproducibility
- Combo hashing for cache lookups
"""

from typing import List, Dict, Any, Optional, Tuple
from .registry import AugmenterRegistry
from .seeds import SeedManager
import pandas as pd


class AugmentationPipeline:
    """
    Pipeline for applying a sequence of augmenters to text data.
    
    Attributes:
        registry: AugmenterRegistry instance
        combo: List of augmenter names in order
        params: Dictionary of parameters for each augmenter
        seed_manager: SeedManager for deterministic augmentation
    """
    
    def __init__(
        self,
        combo: List[str],
        params: Optional[Dict[str, Dict[str, Any]]] = None,
        registry: Optional[AugmenterRegistry] = None,
        seed: int = 13,
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            combo: List of augmenter names to apply in order
            params: Optional parameters for each augmenter (uses defaults if None)
            registry: AugmenterRegistry instance (creates new if None)
            seed: Global seed for reproducibility
        """
        self.combo = combo
        self.params = params or {}
        self.registry = registry or AugmenterRegistry()
        self.seed_manager = SeedManager(seed)
        
        # Validate combo
        self._validate_combo()
        
        # Initialize augmenters
        self.augmenters = self._initialize_augmenters()
    
    def _validate_combo(self) -> None:
        """Validate that all augmenters in combo exist."""
        available = self.registry.list_augmenters()
        for name in self.combo:
            if name not in available:
                raise ValueError(f"Unknown augmenter in combo: {name}")
    
    def _initialize_augmenters(self) -> List[Any]:
        """Initialize augmenter instances for the pipeline."""
        augmenters = []
        
        for i, name in enumerate(self.combo):
            # Get parameters for this augmenter
            aug_params = self.params.get(name, None)
            
            # Get deterministic seed for this position
            aug_seed = self.seed_manager.get_augmenter_seed(i)
            
            # Instantiate augmenter
            augmenter = self.registry.instantiate_augmenter(
                name,
                params=aug_params,
                seed=aug_seed,
            )
            
            augmenters.append(augmenter)
        
        return augmenters
    
    def augment_text(self, text: str, example_idx: int = 0) -> str:
        """
        Augment a single text through the pipeline.
        
        Args:
            text: Input text
            example_idx: Index of example (for deterministic seeding)
            
        Returns:
            Augmented text
        """
        augmented = text
        
        for i, augmenter in enumerate(self.augmenters):
            # Set example-specific seed
            example_seed = self.seed_manager.get_example_seed(example_idx, i)
            self.seed_manager.set_random_seed(example_seed)
            
            # Apply augmenter
            # Note: Actual seeding depends on library API
            try:
                result = augmenter.augment(augmented, n=1)
                if isinstance(result, list):
                    if not result:
                        raise ValueError("Augmenter returned empty list")
                    augmented = result[0]
                else:
                    augmented = result
            except Exception as e:
                # If augmentation fails, log and return original
                print(f"Warning: Augmentation failed for {self.combo[i]}: {e}")
                return text
        
        return augmented
    
    def augment_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str = "evidence_sentence",
        batch_size: int = 1000,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Augment all texts in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_field: Name of text column to augment
            batch_size: Process in batches (for progress tracking)
            verbose: Print progress
            
        Returns:
            DataFrame with augmented texts
        """
        from tqdm import tqdm
        
        # Make a copy
        df_aug = df.copy()
        
        # Augment each text
        augmented_texts = []
        
        iterator = range(len(df))
        if verbose:
            iterator = tqdm(iterator, desc="Augmenting")
        
        for idx in iterator:
            text = df.iloc[idx][text_field]
            augmented = self.augment_text(text, example_idx=idx)
            augmented_texts.append(augmented)
        
        df_aug[text_field] = augmented_texts
        
        return df_aug
    
    def get_combo_hash(self) -> str:
        """
        Get deterministic hash for this combo + params.
        
        Returns:
            Hexadecimal hash string
        """
        from ..utils.hashing import compute_combo_hash
        
        return compute_combo_hash(
            combo=self.combo,
            params=self.params,
            seed=self.seed_manager.global_seed,
        )
    
    def get_combo_info(self) -> Dict[str, Any]:
        """
        Get metadata about this combo.
        
        Returns:
            Dictionary with combo information
        """
        stages = [self.registry.get_augmenter_stage(name) for name in self.combo]
        libs = [self.registry.get_augmenter_lib(name) for name in self.combo]
        
        return {
            "combo": self.combo,
            "k": len(self.combo),
            "stages": stages,
            "unique_stages": list(set(stages)),
            "libraries": libs,
            "params": self.params,
            "hash": self.get_combo_hash(),
        }
    
    def __repr__(self) -> str:
        """String representation of pipeline."""
        return f"AugmentationPipeline({' -> '.join(self.combo)})"
    
    def __len__(self) -> int:
        """Length of pipeline (number of augmenters)."""
        return len(self.combo)
