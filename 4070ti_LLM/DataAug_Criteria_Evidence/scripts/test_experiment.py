#!/usr/bin/env python
"""Test experiment tracking functionality (Phase 15).

Quick test to validate experiment tracking, versioning, reproducibility,
and comparison tools.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import mlflow  # noqa: E402


def test_experiment_tracker() -> None:
    """Test ExperimentTracker."""
    print("\n" + "=" * 80)
    print("TEST 1: Experiment Tracker")
    print("=" * 80)

    from psy_agents_noaug.experiment import ExperimentTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"

        # Create tracker
        tracker = ExperimentTracker(
            experiment_name="test_experiment",
            tracking_uri=tracking_uri,
        )

        print(f"\n✓ Created tracker (experiment={tracker.experiment_name})")

        # Start run
        run_id = tracker.start_run(
            run_name="test_run",
            tags={"test": "true"},
            description="Test run for Phase 15",
        )

        print(f"✓ Started run: {run_id}")

        # Log config
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        }
        tracker.log_config(config)
        print("✓ Logged configuration")

        # Log parameters
        params = {
            "optimizer": "adam",
            "dropout": 0.2,
        }
        tracker.log_parameters(params)
        print(f"✓ Logged {len(params)} parameters")

        # Log metrics
        metrics = {
            "train_loss": 0.5,
            "val_loss": 0.6,
            "val_accuracy": 0.85,
        }
        tracker.log_metrics(metrics)
        print(f"✓ Logged {len(metrics)} metrics")

        # End run
        metadata = tracker.end_run(status="FINISHED")
        print(f"✓ Ended run (status={metadata.status})")

        # Get run info
        run_info = tracker.get_run_info(run_id)
        print(f"✓ Retrieved run info: {len(run_info['metrics'])} metrics")

    print("\n✅ TEST 1 PASSED: Experiment tracker working correctly")


def test_config_versioning() -> None:
    """Test ConfigVersioner."""
    print("\n" + "=" * 80)
    print("TEST 2: Configuration Versioning")
    print("=" * 80)

    from psy_agents_noaug.experiment import ConfigVersioner

    with tempfile.TemporaryDirectory() as tmpdir:
        versioner = ConfigVersioner(storage_dir=tmpdir)

        print("\n✓ Created versioner")

        # Save version 1
        config_v1 = {
            "learning_rate": 0.001,
            "batch_size": 32,
        }
        version1 = versioner.save_version(
            config=config_v1,
            description="Initial config",
            tags={"type": "baseline"},
        )

        print(f"✓ Saved version 1: {version1.version_id}")
        print(f"  Hash: {version1.config_hash}")

        # Save version 2
        config_v2 = {
            "learning_rate": 0.0001,  # Changed
            "batch_size": 32,
            "dropout": 0.2,  # Added
        }
        version2 = versioner.save_version(
            config=config_v2,
            description="Improved config",
            tags={"type": "improved"},
            parent_version=version1.version_id,
        )

        print(f"✓ Saved version 2: {version2.version_id}")

        # List versions
        versions = versioner.list_versions()
        print(f"✓ Listed {len(versions)} versions")

        # Compare versions
        comparison = versioner.compare_versions(
            version1.version_id,
            version2.version_id,
        )

        print("✓ Compared versions:")
        print(f"  Added: {comparison['n_added']} parameters")
        print(f"  Modified: {comparison['n_modified']} parameters")
        print(f"  Unchanged: {comparison['n_unchanged']} parameters")

        # Restore version
        restored_config = versioner.restore_version(version1.version_id)
        assert restored_config == config_v1
        print("✓ Restored version 1")

        # Get latest version
        latest = versioner.get_latest_version()
        assert latest.version_id == version2.version_id
        print(f"✓ Retrieved latest version: {latest.version_id}")

    print("\n✅ TEST 2 PASSED: Configuration versioning working correctly")


def test_reproducibility_manager() -> None:
    """Test ReproducibilityManager."""
    print("\n" + "=" * 80)
    print("TEST 3: Reproducibility Manager")
    print("=" * 80)

    from psy_agents_noaug.experiment import ReproducibilityManager

    manager = ReproducibilityManager(seed=42, deterministic=True)

    print("\n✓ Created reproducibility manager (seed=42)")

    # Set seeds
    manager.set_seeds()
    print("✓ Set all random seeds")

    # Capture snapshot
    snapshot = manager.capture_snapshot()
    print("✓ Captured reproducibility snapshot")
    print(f"  Seed: {snapshot.seeds['main_seed']}")
    print(f"  PyTorch: {snapshot.torch_version}")
    print(f"  NumPy: {snapshot.numpy_version}")
    print(f"  Python: {snapshot.python_version}")
    print(f"  Deterministic: {snapshot.deterministic_mode}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save snapshot
        snapshot_path = Path(tmpdir) / "snapshot.json"
        manager.save_snapshot(snapshot_path)
        print(f"✓ Saved snapshot to: {snapshot_path}")

        # Load snapshot
        loaded_snapshot = manager.load_snapshot(snapshot_path)
        assert loaded_snapshot.seeds == snapshot.seeds
        print("✓ Loaded snapshot successfully")

        # Validate reproducibility
        validation = manager.validate_reproducibility(snapshot_path)
        print("✓ Validated reproducibility:")
        print(f"  Valid: {validation['is_valid']}")
        print(f"  Issues: {len(validation['issues'])}")
        print(f"  Warnings: {len(validation['warnings'])}")

        # Compute hash
        snapshot_hash = manager.compute_snapshot_hash(snapshot)
        print(f"✓ Computed snapshot hash: {snapshot_hash}")

    print("\n✅ TEST 3 PASSED: Reproducibility manager working correctly")


def test_experiment_comparator() -> None:
    """Test ExperimentComparator."""
    print("\n" + "=" * 80)
    print("TEST 4: Experiment Comparator")
    print("=" * 80)

    from psy_agents_noaug.experiment import ExperimentComparator, ExperimentTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"

        # Create multiple experiments
        tracker = ExperimentTracker(
            experiment_name="comparison_test",
            tracking_uri=tracking_uri,
        )

        run_ids = []

        for i in range(3):
            run_id = tracker.start_run(run_name=f"run_{i}")
            run_ids.append(run_id)

            # Log different metrics
            metrics = {
                "val_accuracy": 0.80 + i * 0.05,
                "val_loss": 0.5 - i * 0.1,
            }
            tracker.log_metrics(metrics)

            # Log parameters
            params = {
                "learning_rate": 0.001 * (i + 1),
                "batch_size": 32,
            }
            tracker.log_parameters(params)

            tracker.end_run()

        print(f"\n✓ Created {len(run_ids)} test runs")

        # Create comparator
        comparator = ExperimentComparator(tracking_uri=tracking_uri)
        print("✓ Created comparator")

        # Compare experiments
        comparison = comparator.compare(
            run_ids=run_ids,
            ranking_metric="val_accuracy",
        )

        print("✓ Compared experiments:")
        print(f"  Total experiments: {comparison.summary['n_experiments']}")
        print(f"  Best run: {comparison.summary['best_run'][:8]}")
        print(f"  Best score: {comparison.summary['best_score']:.4f}")

        # Check metrics comparison
        print(f"✓ Metrics comparison shape: {comparison.metrics_comparison.shape}")

        # Check parameters comparison
        print(f"✓ Parameters comparison shape: {comparison.params_comparison.shape}")

        # Check best metrics
        print(f"✓ Best metrics found: {len(comparison.best_metrics)}")
        for metric, info in comparison.best_metrics.items():
            print(f"  {metric}: {info['value']:.4f} (run {info['run_id'][:8]})")

        # Check ranking
        print(f"✓ Ranking generated: {len(comparison.ranking)} experiments")

        # Generate report
        report_path = Path(tmpdir) / "comparison_report.md"
        comparator.generate_comparison_report(comparison, report_path)
        assert report_path.exists()
        print(f"✓ Generated comparison report: {report_path}")

    print("\n✅ TEST 4 PASSED: Experiment comparator working correctly")


def main() -> None:
    """Run all tests."""
    print("=" * 80)
    print("SUPERMAX Phase 15: Experiment Tracking Tests")
    print("=" * 80)

    try:
        test_experiment_tracker()
        test_config_versioning()
        test_reproducibility_manager()
        test_experiment_comparator()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nExperiment tracking functionality is working correctly!")
        print("You can now use:")
        print("  - ExperimentTracker for comprehensive experiment tracking")
        print("  - ConfigVersioner for configuration version control")
        print("  - ReproducibilityManager for reproducibility guarantees")
        print("  - ExperimentComparator for comparing multiple experiments")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
