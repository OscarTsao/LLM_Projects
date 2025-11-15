"""
Combination enumeration for augmentation pipelines.

Generates all valid combinations up to k_max, respecting:
- Exclusion constraints (mutually exclusive augmenters)
- Safety caps (maximum combinations per k)
- Stage diversity requirements
"""

from typing import List, Dict, Set, Tuple, Optional
from itertools import combinations, permutations
from .registry import AugmenterRegistry
import yaml
from pathlib import Path


class ComboGenerator:
    """
    Generator for valid augmentation combinations.
    
    Applies constraints from config to enumerate all valid combinations
    of augmenters up to length k_max.
    
    Attributes:
        registry: AugmenterRegistry instance
        k_max: Maximum pipeline length
        exclusions: List of mutually exclusive augmenter pairs
        safety_caps: Maximum combinations allowed per k
        min_stage_diversity: Minimum number of unique stages required
    """
    
    def __init__(
        self,
        registry: Optional[AugmenterRegistry] = None,
        config_path: str = "configs/run.yaml",
    ):
        """
        Initialize combo generator from config.
        
        Args:
            registry: AugmenterRegistry instance (creates new if None)
            config_path: Path to run.yaml configuration
        """
        self.registry = registry or AugmenterRegistry()
        self.config = self._load_config(config_path)
        
        # Extract combination settings
        combo_config = self.config["combinations"]
        self.k_max = combo_config["k_max"]
        self.exclusions = combo_config.get("exclusions", [])
        self.safety_caps = combo_config.get("safety_caps", {})
        self.min_stage_diversity = combo_config.get("min_stage_diversity", 1)
        
        # Build exclusion set for fast lookup
        self.exclusion_set = self._build_exclusion_set()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _build_exclusion_set(self) -> Set[Tuple[str, str]]:
        """Build set of excluded augmenter pairs."""
        exclusion_set = set()
        
        for pair in self.exclusions:
            if len(pair) != 2:
                raise ValueError(f"Exclusion must be a pair: {pair}")
            
            # Add both orders
            exclusion_set.add(tuple(sorted(pair)))
        
        return exclusion_set
    
    def is_valid_combo(self, combo: List[str]) -> bool:
        """
        Check if a combination is valid.
        
        Args:
            combo: List of augmenter names
            
        Returns:
            True if valid, False otherwise
        """
        # Check exclusions
        for i, aug1 in enumerate(combo):
            for aug2 in combo[i+1:]:
                if tuple(sorted([aug1, aug2])) in self.exclusion_set:
                    return False
        
        # Check stage diversity for k >= 2
        if len(combo) >= 2:
            stages = [self.registry.get_augmenter_stage(aug) for aug in combo]
            unique_stages = len(set(stages))
            
            if unique_stages < self.min_stage_diversity:
                return False
        
        return True
    
    def generate_k_combos(
        self,
        k: int,
        ordered: bool = True,
    ) -> List[List[str]]:
        """
        Generate all valid combinations of length k.
        
        Args:
            k: Combination length
            ordered: If True, treat order as significant (permutations)
                    If False, generate unordered combinations
            
        Returns:
            List of valid combinations
        """
        all_augmenters = self.registry.list_augmenters()
        
        # Generate candidate combinations
        if ordered:
            candidates = permutations(all_augmenters, k)
        else:
            candidates = combinations(all_augmenters, k)
        
        # Filter valid combinations
        valid = []
        for combo in candidates:
            combo_list = list(combo)
            if self.is_valid_combo(combo_list):
                valid.append(combo_list)
        
        # Apply safety cap
        safety_key = f"max_{'single' if k == 1 else 'pairs' if k == 2 else 'triples'}"
        max_combos = self.safety_caps.get(safety_key, len(valid))
        
        return valid[:max_combos]
    
    def generate_all_combos(
        self,
        ordered: bool = True,
        verbose: bool = True,
    ) -> Dict[int, List[List[str]]]:
        """
        Generate all valid combinations up to k_max.
        
        Args:
            ordered: If True, treat order as significant
            verbose: Print statistics
            
        Returns:
            Dictionary mapping k to list of combinations
        """
        all_combos = {}
        
        for k in range(1, self.k_max + 1):
            combos = self.generate_k_combos(k, ordered=ordered)
            all_combos[k] = combos
            
            if verbose:
                print(f"Generated {len(combos)} valid combinations for k={k}")
        
        if verbose:
            total = sum(len(combos) for combos in all_combos.values())
            print(f"\nTotal valid combinations: {total}")
        
        return all_combos
    
    def get_combo_statistics(
        self,
        combos: Dict[int, List[List[str]]],
    ) -> Dict[str, any]:
        """
        Compute statistics about generated combinations.
        
        Args:
            combos: Dictionary mapping k to list of combinations
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_combos": sum(len(c) for c in combos.values()),
            "by_k": {k: len(c) for k, c in combos.items()},
        }
        
        # Stage distribution
        stage_counts = {}
        for k, combo_list in combos.items():
            for combo in combo_list:
                stages = tuple(sorted(set(
                    self.registry.get_augmenter_stage(aug) for aug in combo
                )))
                stage_counts[stages] = stage_counts.get(stages, 0) + 1
        
        stats["stage_distributions"] = stage_counts
        
        # Most common augmenters
        augmenter_counts = {}
        for k, combo_list in combos.items():
            for combo in combo_list:
                for aug in combo:
                    augmenter_counts[aug] = augmenter_counts.get(aug, 0) + 1
        
        stats["augmenter_frequency"] = dict(
            sorted(augmenter_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return stats
    
    def save_combos(
        self,
        combos: Dict[int, List[List[str]]],
        output_path: str,
    ) -> None:
        """
        Save combinations to JSON file.
        
        Args:
            combos: Dictionary mapping k to list of combinations
            output_path: Path to output JSON file
        """
        import json
        
        # Convert to serializable format
        serializable = {
            str(k): [list(combo) for combo in combo_list]
            for k, combo_list in combos.items()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        
        print(f"Saved {sum(len(c) for c in combos.values())} combinations to {output_path}")
    
    def load_combos(self, input_path: str) -> Dict[int, List[List[str]]]:
        """
        Load combinations from JSON file.
        
        Args:
            input_path: Path to input JSON file
            
        Returns:
            Dictionary mapping k to list of combinations
        """
        import json
        
        with open(input_path) as f:
            serializable = json.load(f)
        
        # Convert back to original format
        combos = {
            int(k): [list(combo) for combo in combo_list]
            for k, combo_list in serializable.items()
        }
        
        return combos
