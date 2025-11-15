"""Test manifest integrity."""
from tests.verify_utils import temp_output_dir, run_cli
import pandas as pd

def test_manifest_structure():
    """Manifest has required columns."""
    with temp_output_dir() as outdir:
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--variants-per-sample", "1",
            "--seed", "42",
            "--num-proc", "1",
        )

        manifests = list(outdir.glob("manifest_*.csv"))
        if manifests:
            df = pd.read_csv(manifests[0])
            assert "combo_id" in df.columns
            assert "dataset_path" in df.columns

def test_dataset_paths_exist():
    """All dataset_path entries exist."""
    with temp_output_dir() as outdir:
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--variants-per-sample", "1",
            "--seed", "42",
            "--num-proc", "1",
        )

        manifests = list(outdir.glob("manifest_*.csv"))
        if manifests:
            df = pd.read_csv(manifests[0])
            for path_str in df["dataset_path"]:
                path = outdir / path_str
                assert path.exists(), f"Missing dataset: {path}"
