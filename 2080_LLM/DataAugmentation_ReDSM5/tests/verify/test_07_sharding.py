"""Test sharding correctness."""
from tests.verify_utils import temp_output_dir, run_cli
import pandas as pd

def test_sharding_no_overlap():
    """Shards don't overlap."""
    shard_outputs = []
    for shard_idx in range(2):
        with temp_output_dir() as outdir:
            run_cli(
                "--input", "tests/fixtures/mini_annotations.csv",
                "--output-root", str(outdir),
                "--combo-mode", "singletons",
                "--num-shards", "2",
                "--shard-index", str(shard_idx),
                "--variants-per-sample", "1",
                "--seed", "42",
                "--num-proc", "1",
            )
            manifests = list(outdir.glob("manifest_*.csv"))
            if manifests:
                df = pd.read_csv(manifests[0])
                shard_outputs.append(set(df["combo_id"]))

    if len(shard_outputs) == 2:
        overlap = shard_outputs[0] & shard_outputs[1]
        assert len(overlap) == 0, f"Shards overlap: {overlap}"
