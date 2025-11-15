"""
Cache size estimation utilities.

Helps estimate disk space requirements before generating full cache.
"""

from typing import Dict, List
import pandas as pd


def estimate_cache_size(
    sample_df: pd.DataFrame,
    num_combos: int,
    splits: List[str] = None,
    split_sizes: Dict[str, int] = None,
    compression_ratio: float = 0.3,
) -> Dict[str, float]:
    """
    Estimate total cache size.
    
    Args:
        sample_df: Sample DataFrame to estimate row size
        num_combos: Total number of combinations
        splits: List of split names
        split_sizes: Dictionary mapping split names to sizes
        compression_ratio: Expected compression ratio (0-1)
        
    Returns:
        Dictionary with size estimates
    """
    splits = splits or ["train", "val", "test"]
    
    # Estimate uncompressed size per row
    sample_parquet = sample_df.head(1000).to_parquet()
    bytes_per_row = len(sample_parquet) / 1000
    
    # Estimate total size
    total_rows = 0
    split_estimates = {}
    
    for split in splits:
        if split_sizes and split in split_sizes:
            rows = split_sizes[split]
        else:
            rows = len(sample_df)  # Fallback to sample size
        
        total_rows += rows
        split_estimates[split] = rows * bytes_per_row
    
    # Account for all combos
    total_uncompressed = total_rows * bytes_per_row * num_combos
    total_compressed = total_uncompressed * compression_ratio
    
    return {
        "bytes_per_row": bytes_per_row,
        "total_rows": total_rows,
        "num_combos": num_combos,
        "total_uncompressed_bytes": total_uncompressed,
        "total_compressed_bytes": total_compressed,
        "total_uncompressed_gb": total_uncompressed / (1024**3),
        "total_compressed_gb": total_compressed / (1024**3),
        "split_estimates": split_estimates,
    }


def format_size(bytes: float) -> str:
    """
    Format byte size in human-readable format.
    
    Args:
        bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.23 GB")
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes)
    unit_idx = 0
    
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    
    return f"{size:.2f} {units[unit_idx]}"


def print_cache_estimate(estimates: Dict[str, float]) -> None:
    """
    Print cache size estimates in readable format.
    
    Args:
        estimates: Dictionary from estimate_cache_size
    """
    print("\n=== Cache Size Estimates ===")
    print(f"Bytes per row: {estimates['bytes_per_row']:.2f}")
    print(f"Total rows: {estimates['total_rows']:,}")
    print(f"Number of combos: {estimates['num_combos']:,}")
    print(f"\nUncompressed: {format_size(estimates['total_uncompressed_bytes'])}")
    print(f"Compressed: {format_size(estimates['total_compressed_bytes'])}")
    print(f"\nSplit estimates:")
    for split, size in estimates['split_estimates'].items():
        print(f"  {split}: {format_size(size)}")
    print("=" * 30 + "\n")


def estimate_hpo_time(
    num_trials: int,
    avg_trial_time_minutes: float,
    parallel_trials: int = 1,
) -> Dict[str, float]:
    """
    Estimate total HPO time.
    
    Args:
        num_trials: Number of trials
        avg_trial_time_minutes: Average time per trial in minutes
        parallel_trials: Number of parallel trials
        
    Returns:
        Dictionary with time estimates
    """
    total_trial_time = num_trials * avg_trial_time_minutes
    wallclock_time = total_trial_time / parallel_trials
    
    return {
        "num_trials": num_trials,
        "avg_trial_time_minutes": avg_trial_time_minutes,
        "parallel_trials": parallel_trials,
        "total_trial_time_hours": total_trial_time / 60,
        "wallclock_time_hours": wallclock_time / 60,
    }
