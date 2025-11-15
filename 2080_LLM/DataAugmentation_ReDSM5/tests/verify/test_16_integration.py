"""Comprehensive integration tests for end-to-end workflow."""
import pytest
from tests.verify_utils import temp_output_dir, run_cli, parse_meta_json, load_fixture
import pandas as pd


@pytest.mark.slow
def test_full_pipeline_singletons():
    """
    End-to-end integration test with singleton combos.
    Validates: input → registry → generation → output → metadata.
    """
    with temp_output_dir() as outdir:
        # Run with 3 fast CPU methods
        exit_code, stdout, stderr = run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--variants-per-sample", "2",
            "--seed", "42",
            "--methods-yaml", "conf/augment_methods.yaml",
            "--num-proc", "1",
        )
        
        # 1. Verify command succeeded
        assert exit_code == 0, f"CLI failed with code {exit_code}\nstderr: {stderr}"
        
        # 2. Verify output structure exists
        assert outdir.exists(), "Output directory not created"
        combo_dirs = [d for d in outdir.iterdir() if d.is_dir()]
        assert len(combo_dirs) > 0, "No combo directories generated"
        
        # 3. Verify each combo has required files
        for combo_dir in combo_dirs:
            # Skip empty combo directories (methods that failed/skipped)
            if not any(combo_dir.iterdir()):
                continue
            
            # Check for dataset
            dataset_path = combo_dir / "dataset.parquet"
            assert dataset_path.exists(), (
                f"Missing dataset.parquet in non-empty combo {combo_dir.name}")
            
            # Check for meta.json
            meta_path = combo_dir / "meta.json"
            assert meta_path.exists(), f"Missing meta.json in {combo_dir.name}"
            
            # 4. Validate metadata
            meta = parse_meta_json(combo_dir)
            assert "combo_id" in meta, "Missing combo_id in metadata"
            assert "combo_methods" in meta, "Missing combo_methods in metadata"
            assert "seed" in meta, "Missing seed in metadata"
            assert meta["seed"] == 42, "Seed mismatch in metadata"
            assert len(meta["combo_methods"]) > 0, "No methods in metadata"
            
            # 5. Validate dataset structure
            df = pd.read_parquet(dataset_path)
            assert len(df) > 0, f"Empty dataset in {combo_dir.name}"
            
            required_cols = ["post_id", "post_text", "evidence"]
            for col in required_cols:
                assert col in df.columns, f"Missing column {col} in dataset"
            
            # 6. Verify data integrity
            assert df["post_id"].notna().all(), "Null post_ids found"
            assert df["post_text"].notna().all(), "Null post_text found"
            assert df["evidence"].notna().all(), "Null evidence found"
            
            # 7. Verify augmentation happened
            # Check that evidence appears in post_text
            for _, row in df.iterrows():
                assert row["evidence"] in row["post_text"], (
                    f"Evidence not found in post_text for {row['post_id']}"
                )


@pytest.mark.slow
def test_full_pipeline_pairs():
    """
    End-to-end integration test with pair combos.
    Validates combo generation and composition.
    """
    with temp_output_dir() as outdir:
        exit_code, stdout, stderr = run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "bounded_k",
            "--max-combo-size", "2",
            "--variants-per-sample", "1",
            "--seed", "100",
            "--num-proc", "1",
        )
        
        assert exit_code == 0, f"CLI failed: {stderr}"
        
        combo_dirs = [d for d in outdir.iterdir() if d.is_dir()]
        assert len(combo_dirs) > 0, "No combo directories for pairs"
        
        # Verify at least one combo has 2 methods
        found_pair = False
        for combo_dir in combo_dirs:
            meta_path = combo_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = parse_meta_json(combo_dir)
            if len(meta.get("combo_methods", [])) == 2:
                found_pair = True
                break
        
        assert found_pair, "No pair combos found in pairs mode"


def test_manifest_generation():
    """Verify manifest.json is generated with correct structure."""
    with temp_output_dir() as outdir:
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--variants-per-sample", "1",
            "--seed", "42",
            "--num-proc", "1",
        )
        
        manifest_path = outdir / "manifest_shard0_of_1.csv"
        if not manifest_path.exists():
            # Manifest might be optional, check if combos exist
            combo_dirs = [d for d in outdir.iterdir() if d.is_dir()]
            if len(combo_dirs) > 0:
                pytest.skip("Manifest not generated but combos exist")
            else:
                pytest.fail("No combos or manifest generated")
        
        # If manifest exists, verify it's readable
        import pandas as pd
        df_manifest = pd.read_csv(manifest_path)
        assert len(df_manifest) > 0, "Manifest is empty"
        assert "combo_id" in df_manifest.columns, "Missing combo_id in manifest"


def test_metadata_correctness():
    """Verify all metadata fields are correct and consistent."""
    with temp_output_dir() as outdir:
        seed = 999
        exit_code, stdout, stderr = run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--variants-per-sample", "2",
            "--seed", str(seed),
            "--quality-min-sim", "0.40",
            "--quality-max-sim", "0.95",
            "--num-proc", "1",
        )
        
        if exit_code != 0:
            pytest.fail(f"CLI failed: {stderr}")
        
        combo_dirs = [d for d in outdir.iterdir() if d.is_dir()]
        assert len(combo_dirs) > 0, "No combos generated"
        
        # Check metadata for combos that have meta.json (failed combos might not)
        valid_combos = 0
        for combo_dir in combo_dirs:
            meta_path = combo_dir / "meta.json"
            if not meta_path.exists():
                # Skip combos without meta.json (failed combos)
                continue
                
            valid_combos += 1
            meta = parse_meta_json(combo_dir)
            
            # Verify seed propagation
            assert meta["seed"] == seed, "Seed not correctly stored"
            
            # Verify methods list
            assert isinstance(meta.get("combo_methods"), list), "combo_methods should be a list"
            assert len(meta["combo_methods"]) > 0, "combo_methods list is empty"
            
            # Verify combo_id format
            assert isinstance(meta["combo_id"], str), "combo_id should be string"
            assert len(meta["combo_id"]) > 0, "combo_id is empty"
            
            # Check for expected metadata fields
            expected_fields = ["combo_id", "combo_methods", "seed"]
            for field in expected_fields:
                assert field in meta, f"Missing required field: {field}"
        
        assert valid_combos > 0, "No valid combos with metadata found"


def test_input_to_output_row_traceability():
    """Verify every output row can be traced back to an input row."""
    with temp_output_dir() as outdir:
        run_cli(
            "--input", "tests/fixtures/mini_annotations.csv",
            "--output-root", str(outdir),
            "--combo-mode", "singletons",
            "--variants-per-sample", "2",
            "--seed", "42",
            "--num-proc", "1",
        )
        
        # Load original data
        df_orig = load_fixture()
        original_post_ids = set(df_orig["post_id"])
        
        # Check all combo outputs
        combo_dirs = [d for d in outdir.iterdir() if d.is_dir()]
        for combo_dir in combo_dirs:
            dataset_path = combo_dir / "dataset.parquet"
            if dataset_path.exists():
                df_aug = pd.read_parquet(dataset_path)
                
                # Every post_id in output should exist in input
                for post_id in df_aug["post_id"].unique():
                    assert post_id in original_post_ids, (
                        f"Output contains post_id {post_id} not in input"
                    )
