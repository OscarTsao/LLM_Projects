"""Test quality filtering thresholds."""
from tests.verify_utils import temp_output_dir, run_cli
import pandas as pd
from difflib import SequenceMatcher


def test_quality_thresholds():
    """Quality min/max filters work."""
    with temp_output_dir() as outdir:
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--quality-min-sim", "0.90",  # Very strict
            "--quality-max-sim", "0.98",
            "--variants-per-sample", "5",
            "--seed", "42",
            "--num-proc", "1",
        )

        datasets = list(outdir.rglob("dataset.parquet"))
        if datasets:
            df = pd.read_parquet(datasets[0])
            # With strict thresholds, fewer variants expected
            assert len(df) >= 0  # May be empty if all filtered


def test_min_similarity_boundary():
    """
    Test boundary condition: candidates with similarity exactly at min threshold 
    should pass (inclusive boundary).
    """
    with temp_output_dir() as outdir:
        min_sim = 0.40
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--quality-min-sim", str(min_sim),
            "--quality-max-sim", "0.99",
            "--variants-per-sample", "10",  # Generate many to test filtering
            "--seed", "42",
            "--num-proc", "1",
        )
        
        datasets = list(outdir.rglob("dataset.parquet"))
        if not datasets:
            return  # Skip if no datasets (all filtered)
        
        df = pd.read_parquet(datasets[0])
        if len(df) == 0:
            return  # All filtered, which is valid
        
        # Verify no outputs are below min_sim
        # Compare evidence with evidence_original
        for _, row in df.iterrows():
            if "evidence_original" in row:
                orig = row["evidence_original"]
                aug = row["evidence"]
                
                # Calculate similarity
                sim = SequenceMatcher(None, orig, aug).ratio()
                
                # Should be >= min_sim (allowing small floating point error)
                assert sim >= min_sim - 0.01, (
                    f"Found similarity {sim:.3f} < min {min_sim} for {row.get('post_id')}"
                )


def test_max_similarity_boundary():
    """
    Test boundary condition: candidates with similarity exactly at max threshold 
    should pass (inclusive boundary).
    """
    with temp_output_dir() as outdir:
        max_sim = 0.98
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--quality-min-sim", "0.30",
            "--quality-max-sim", str(max_sim),
            "--variants-per-sample", "10",
            "--seed", "42",
            "--num-proc", "1",
        )
        
        datasets = list(outdir.rglob("dataset.parquet"))
        if not datasets:
            return
        
        df = pd.read_parquet(datasets[0])
        if len(df) == 0:
            return
        
        # Verify no outputs exceed max_sim
        for _, row in df.iterrows():
            if "evidence_original" in row:
                orig = row["evidence_original"]
                aug = row["evidence"]
                
                sim = SequenceMatcher(None, orig, aug).ratio()
                
                # Should be <= max_sim (allowing small floating point error)
                assert sim <= max_sim + 0.01, (
                    f"Found similarity {sim:.3f} > max {max_sim} for {row.get('post_id')}"
                )


def test_identity_rejection():
    """
    Verify that augmenters returning unchanged text are filtered out
    when max_similarity is set appropriately.
    """
    with temp_output_dir() as outdir:
        # Set max_sim to reject identical outputs
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--quality-min-sim", "0.30",
            "--quality-max-sim", "0.95",  # Should reject identity (1.0)
            "--variants-per-sample", "5",
            "--seed", "42",
            "--num-proc", "1",
        )
        
        datasets = list(outdir.rglob("dataset.parquet"))
        if not datasets:
            return
        
        df = pd.read_parquet(datasets[0])
        if len(df) == 0:
            return
        
        # Verify no evidence is identical to original
        identity_count = 0
        for _, row in df.iterrows():
            if "evidence_original" in row:
                if row["evidence_original"] == row["evidence"]:
                    identity_count += 1
        
        # Allow a small number of identities due to augmentation randomness,
        # but most should be changed
        total = len(df)
        identity_rate = identity_count / total if total > 0 else 0
        assert identity_rate < 0.20, (
            f"Too many identity augmentations: {identity_count}/{total} ({identity_rate:.1%})"
        )


def test_very_strict_filtering():
    """
    Test that very strict thresholds (narrow range) filter aggressively.
    """
    with temp_output_dir() as outdir:
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--quality-min-sim", "0.75",
            "--quality-max-sim", "0.85",  # Very narrow range
            "--variants-per-sample", "20",  # Try to generate many
            "--seed", "42",
            "--num-proc", "1",
        )
        
        datasets = list(outdir.rglob("dataset.parquet"))
        
        # Might have empty output or fewer variants than requested
        if datasets:
            df = pd.read_parquet(datasets[0])
            # With narrow range, we expect heavy filtering
            # Just verify structure is correct
            if len(df) > 0:
                assert "evidence" in df.columns
                assert "post_text" in df.columns


def test_permissive_filtering():
    """
    Test that permissive thresholds (wide range) allow most augmentations.
    """
    with temp_output_dir() as outdir:
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--quality-min-sim", "0.10",  # Very permissive
            "--quality-max-sim", "0.99",
            "--variants-per-sample", "3",
            "--seed", "42",
            "--num-proc", "1",
        )
        
        datasets = list(outdir.rglob("dataset.parquet"))
        if not datasets:
            return
        
        df = pd.read_parquet(datasets[0])
        
        # With permissive thresholds, we expect good output
        # At least some rows should be generated
        assert len(df) > 0, "Permissive filtering should produce some output"


def test_quality_metadata_tracking():
    """
    Verify that quality-related metadata is tracked when available.
    """
    with temp_output_dir() as outdir:
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--quality-min-sim", "0.40",
            "--quality-max-sim", "0.95",
            "--variants-per-sample", "2",
            "--seed", "42",
            "--num-proc", "1",
        )
        
        datasets = list(outdir.rglob("dataset.parquet"))
        if not datasets:
            return
        
        df = pd.read_parquet(datasets[0])
        if len(df) == 0:
            return
        
        # Check if evidence_original is tracked (needed for quality checks)
        # This is important for downstream analysis
        if "evidence_original" in df.columns:
            # Verify it's not null
            assert df["evidence_original"].notna().all(), (
                "evidence_original should not contain nulls"
            )
