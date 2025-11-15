"""
Deterministic hashing for augmentation combos and cached data.

Uses xxhash for fast, collision-resistant hashing.
"""

from typing import List, Dict, Any
import xxhash
import json


def compute_combo_hash(
    combo: List[str],
    params: Dict[str, Dict[str, Any]],
    seed: int,
) -> str:
    """
    Compute deterministic hash for an augmentation combination.
    
    Hash is based on:
    - Ordered list of augmenter names
    - Parameters for each augmenter
    - Global seed
    
    Args:
        combo: Ordered list of augmenter names
        params: Dictionary of parameters for each augmenter
        seed: Global seed
        
    Returns:
        Hexadecimal hash string (16 characters)
    """
    # Create canonical representation
    combo_data = {
        "combo": combo,
        "params": params,
        "seed": seed,
    }
    
    # Convert to JSON with sorted keys for determinism
    combo_json = json.dumps(combo_data, sort_keys=True)
    
    # Compute hash
    hasher = xxhash.xxh64()
    hasher.update(combo_json.encode("utf-8"))
    
    return hasher.hexdigest()


def compute_text_hash(text: str) -> str:
    """
    Compute hash of a text string.
    
    Args:
        text: Input text
        
    Returns:
        Hexadecimal hash string (16 characters)
    """
    hasher = xxhash.xxh64()
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def compute_dataset_hash(texts: List[str]) -> str:
    """
    Compute hash of entire dataset.
    
    Args:
        texts: List of text strings
        
    Returns:
        Hexadecimal hash string
    """
    hasher = xxhash.xxh64()
    
    for text in texts:
        hasher.update(text.encode("utf-8"))
    
    return hasher.hexdigest()


def verify_cache_integrity(
    combo_hash: str,
    combo: List[str],
    params: Dict[str, Dict[str, Any]],
    seed: int,
) -> bool:
    """
    Verify that cached data matches the expected combo.
    
    Args:
        combo_hash: Hash from cached file
        combo: Combo to verify
        params: Parameters to verify
        seed: Seed to verify
        
    Returns:
        True if hash matches, False otherwise
    """
    expected_hash = compute_combo_hash(combo, params, seed)
    return combo_hash == expected_hash


def parse_cache_filename(filename: str) -> Dict[str, str]:
    """
    Parse cache filename to extract metadata.
    
    Expected format: aug_{combo_hash}_{split}.parquet
    
    Args:
        filename: Cache filename
        
    Returns:
        Dictionary with 'combo_hash' and 'split'
    """
    import re
    
    pattern = r"aug_([0-9a-f]+)_(\w+)\.parquet"
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Invalid cache filename format: {filename}")
    
    return {
        "combo_hash": match.group(1),
        "split": match.group(2),
    }


def generate_cache_filename(combo_hash: str, split: str) -> str:
    """
    Generate cache filename from hash and split.
    
    Args:
        combo_hash: Combo hash
        split: Dataset split (train, val, test)
        
    Returns:
        Filename in standard format
    """
    return f"aug_{combo_hash}_{split}.parquet"
