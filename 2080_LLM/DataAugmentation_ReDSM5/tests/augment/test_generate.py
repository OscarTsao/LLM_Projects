from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tools.generate_augsets import main as generate_main


def _write_methods_yaml(path: Path) -> Path:
    config = {
        "methods": [
            {
                "id": "stub_prefix",
                "lib": "nlpaug",
                "kind": "tests.augment._stubs.PrefixAugmenter",
                "args": {},
                "requires_gpu": False,
            },
            {
                "id": "stub_suffix",
                "lib": "nlpaug",
                "kind": "tests.augment._stubs.RandomSuffixAugmenter",
                "args": {"suffixes": ["!", "?", "..."]},
                "requires_gpu": False,
            },
            {
                "id": "stub_noisy",
                "lib": "nlpaug",
                "kind": "tests.augment._stubs.NoisyPassthroughAugmenter",
                "args": {},
                "requires_gpu": False,
            },
        ]
    }
    yaml_path = path / "methods.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path


def _write_input_csv(path: Path) -> Path:
    data = [
        {
            "post_text": "Alpha sentence. Evidence one appears here.",
            "evidence": "Evidence one appears here.",
            "post_id": "row1",
        },
        {
            "post_text": "Intro. Second evidence snippet lives here!",
            "evidence": "Second evidence snippet lives here!",
            "post_id": "row2",
        },
    ]
    df = pd.DataFrame(data)
    csv_path = path / "annotations.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _snapshot_outputs(output_root: Path) -> Dict[str, List[tuple]]:
    results: Dict[str, List[tuple]] = {}
    for combo_dir in sorted(output_root.glob("combo_*")):
        dataset_files = list(combo_dir.glob("dataset.*"))
        assert dataset_files, f"No dataset written for combo {combo_dir}"
        dataset = dataset_files[0]
        if dataset.suffix == ".csv":
            df = pd.read_csv(dataset)
        else:
            df = pd.read_parquet(dataset)
        df = df.sort_values(by=["post_text", "evidence", "source_combo"]).reset_index(drop=True)
        records = [tuple(row) for row in df[["post_text", "evidence", "source_combo"]].itertuples(index=False, name=None)]
        results[combo_dir.name] = records
    return results


def _run_cli(args: Sequence[str]) -> None:
    generate_main(list(args))


def test_combo_determinism(tmp_path: Path) -> None:
    methods_yaml = _write_methods_yaml(tmp_path)
    csv_path = _write_input_csv(tmp_path)
    output_root = tmp_path / "out"

    def run(seed: int) -> Dict[str, List[tuple]]:
        args = [
            "--input",
            str(csv_path),
            "--text-col",
            "post_text",
            "--evidence-col",
            "evidence",
            "--methods-yaml",
            str(methods_yaml),
            "--output-root",
            str(output_root),
            "--combo-mode",
            "bounded_k",
            "--max-combo-size",
            "2",
            "--variants-per-sample",
            "2",
            "--num-proc",
            "1",
            "--seed",
            str(seed),
            "--save-format",
            "csv",
            "--force",
        ]
        _run_cli(args)
        return _snapshot_outputs(output_root)

    first = run(42)
    second = run(42)
    assert first == second

    third = run(99)
    assert third != first


def test_dedup_and_quality_filter(tmp_path: Path) -> None:
    methods_yaml = _write_methods_yaml(tmp_path)
    csv_path = _write_input_csv(tmp_path)
    output_root = tmp_path / "dedupe"

    args = [
        "--input",
        str(csv_path),
        "--text-col",
        "post_text",
        "--evidence-col",
        "evidence",
        "--methods-yaml",
        str(methods_yaml),
        "--output-root",
        str(output_root),
        "--combo-mode",
        "singletons",
        "--variants-per-sample",
        "2",
        "--num-proc",
        "1",
        "--seed",
        "123",
        "--save-format",
        "csv",
        "--force",
    ]
    _run_cli(args)

    manifest_path = output_root / "manifest_shard0_of_1.csv"
    manifest = pd.read_csv(manifest_path)
    noisy_rows = manifest[manifest["methods"] == "stub_noisy"]
    assert not noisy_rows.empty
    dataset_path = Path(noisy_rows.iloc[0]["dataset_path"])
    df = pd.read_csv(dataset_path)

    assert df["evidence"].str.endswith("***").all()
    assert (df["evidence"].str.casefold() != df["evidence_original"].str.casefold()).all()


def test_resume_skips_existing_outputs(tmp_path: Path) -> None:
    methods_yaml = _write_methods_yaml(tmp_path)
    csv_path = _write_input_csv(tmp_path)
    output_root = tmp_path / "resume"

    base_args = [
        "--input",
        str(csv_path),
        "--text-col",
        "post_text",
        "--evidence-col",
        "evidence",
        "--methods-yaml",
        str(methods_yaml),
        "--output-root",
        str(output_root),
        "--combo-mode",
        "singletons",
        "--variants-per-sample",
        "1",
        "--num-proc",
        "1",
        "--seed",
        "7",
        "--save-format",
        "csv",
    ]

    _run_cli(base_args + ["--force"])

    # Capture dataset modification time.
    combo_dir = next(output_root.glob("combo_*"))
    dataset_path = next(combo_dir.glob("dataset.*"))
    initial_mtime = dataset_path.stat().st_mtime

    time.sleep(1.0)
    _run_cli(base_args)

    second_mtime = dataset_path.stat().st_mtime
    assert second_mtime == pytest.approx(initial_mtime)

    manifest_path = output_root / "manifest_shard0_of_1.csv"
    manifest = pd.read_csv(manifest_path)
    assert (manifest["status"] == "skipped").any()
