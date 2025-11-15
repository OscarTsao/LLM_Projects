"""Verify variant count constraints."""
from tests.verify_utils import temp_output_dir, run_cli
import pandas as pd

def test_variants_per_sample_limit():
    """Variants per sample <= requested count."""
    with temp_output_dir() as outdir:
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--variants-per-sample", "2",
            "--seed", "42",
            "--num-proc", "1",
        )

        datasets = list(outdir.rglob("dataset.parquet"))
        if datasets:
            df = pd.read_parquet(datasets[0])
            # Group by source row identifier
            if "post_id" in df.columns:
                variant_counts = df.groupby("post_id").size()
                assert all(variant_counts <= 2), "Exceeded variants-per-sample limit"
