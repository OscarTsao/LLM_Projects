"""
Deterministic seeding for reproducible augmentation.

Ensures that:
- Same combo + params + seed always produces same output
- Different examples get different random variations
- Pipeline position affects randomness
"""

from typing import List
import random
import numpy as np


class SeedManager:
    """
    Manager for deterministic seeding in augmentation pipelines.
    
    Uses a hierarchical seeding scheme:
    - Global seed: Fixed for entire experiment
    - Augmenter seed: Derived from position in pipeline
    - Example seed: Derived from example index + augmenter position
    
    Attributes:
        global_seed: Root seed for reproducibility
    """
    
    def __init__(self, global_seed: int = 13):
        """
        Initialize seed manager.
        
        Args:
            global_seed: Root seed for all randomness
        """
        self.global_seed = global_seed
    
    def get_augmenter_seed(self, position: int) -> int:
        """
        Get seed for an augmenter at a specific position in pipeline.
        
        Args:
            position: 0-indexed position in pipeline
            
        Returns:
            Deterministic seed for this augmenter
        """
        # Use position-based offset from global seed
        return self.global_seed + position * 1000
    
    def get_example_seed(self, example_idx: int, position: int) -> int:
        """
        Get seed for augmenting a specific example at a specific position.
        
        Args:
            example_idx: Index of example in dataset
            position: Position in pipeline (0-indexed)
            
        Returns:
            Deterministic seed for this example + augmenter
        """
        # Combine global seed, position, and example index
        base_seed = self.get_augmenter_seed(position)
        return base_seed + example_idx
    
    def set_random_seed(self, seed: int) -> None:
        """
        Set Python and NumPy random seeds.
        
        Args:
            seed: Seed value
        """
        random.seed(seed)
        np.random.seed(seed)
    
    def set_all_seeds(self, seed: int) -> None:
        """
        Set all random seeds (Python, NumPy, PyTorch).
        
        Args:
            seed: Seed value
        """
        self.set_random_seed(seed)
        
        # Try to set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
    def get_pipeline_seeds(self, k: int) -> List[int]:
        """
        Get all seeds for a pipeline of length k.
        
        Args:
            k: Length of pipeline
            
        Returns:
            List of k seeds
        """
        return [self.get_augmenter_seed(i) for i in range(k)]
    
    def reset_to_global(self) -> None:
        """Reset all random seeds to global seed."""
        self.set_all_seeds(self.global_seed)
