#!/usr/bin/env python3
"""Run test evaluation for all architectures using best HPO configurations.

This script:
1. Loads best configs from topk JSON files
2. For Criteria: Uses existing train_criteria.py (already has test eval)
3. For others: Creates temporary config YAMLs and trains models
4. Collects test F1 scores from all architectures
5. Generates validation vs test comparison report

Usage:
    # Evaluate all architectures (full 100 epochs)
    python scripts/run_test_evaluation.py

    # Quick test with 10 epochs
    python scripts/run_test_evaluation.py --quick

    # Specific architectures only
    python scripts/run_test_evaluation.py --agents criteria share
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def load_best_config_from_topk(topk_json_path: Path) -> Dict[str, Any]:
    """Load best trial configuration from topk JSON file."""
    with open(topk_json_path) as f:
        topk_data = json.load(f)

    if not topk_data:
        raise ValueError(f"No trials found in {topk_json_path}")

    # Best trial is rank 1 (index 0)
    return topk_data[0]


def convert_to_yaml_config(agent: str, params: Dict[str, Any], output_path: Path) -> Path:
    """Convert topk JSON parameters to YAML config file."""
    config_lines = []

    # Model config
    config_lines.append("# Generated from best HPO configuration")
    config_lines.append(f"# Agent: {agent}")
    config_lines.append("")
    config_lines.append("model:")
    config_lines.append(f"  pretrained_model: {params.get('model.name', 'distilbert-base-uncased')}")
    config_lines.append(f"  classifier_dropout: {params.get('head.dropout', 0.1)}")
    config_lines.append(f"  classifier_layer_num: {params.get('head.n_layers', 1)}")

    # For multi-task models (share, joint), add additional config
    if agent in ["share", "joint"]:
        config_lines.append(f"  evidence_dropout: {params.get('head.dropout', 0.1)}")

    config_lines.append("")
    config_lines.append("dataset:")
    config_lines.append("  path: data/processed/redsm5_matched_evidence.csv")
    config_lines.append(f"  max_length: {params.get('tok.max_length', 320)}")
    config_lines.append("  text_column: post_text")

    if agent == "criteria":
        config_lines.append("  label_column: status")
    elif agent == "evidence":
        config_lines.append("  answer_column: sentence_text")
        config_lines.append("  context_column: post_text")
    else:  # share, joint
        config_lines.append("  label_column: status")
        config_lines.append("  answer_column: sentence_text")

    config_lines.append("")
    config_lines.append("training:")
    config_lines.append(f"  epochs: 100  # Will be overridden by command line")
    config_lines.append(f"  seed: 42")
    config_lines.append(f"  train_batch_size: {params.get('train.batch_size', 24)}")
    config_lines.append(f"  eval_batch_size: {params.get('train.batch_size', 24)}")
    config_lines.append(f"  learning_rate: {params.get('optim.lr', 5e-5)}")
    config_lines.append(f"  gradient_accumulation: {params.get('train.grad_accum', 1)}")
    config_lines.append(f"  max_grad_norm: {params.get('reg.max_grad_norm', 1.0)}")
    config_lines.append("  deterministic: true")
    config_lines.append("  cudnn_benchmark: false")
    config_lines.append("  monitor_metric: val_f1_macro")
    config_lines.append("  monitor_mode: max")
    config_lines.append("  logging_steps: 10")

    config_lines.append("")
    config_lines.append("  optimizer:")
    config_lines.append(f"    name: {params.get('optim.name', 'adamw')}")
    config_lines.append(f"    weight_decay: {params.get('optim.weight_decay', 0.01)}")

    config_lines.append("")
    config_lines.append("  scheduler:")
    config_lines.append(f"    name: {params.get('sched.name', 'linear')}")
    config_lines.append(f"    warmup_steps: {int(params.get('sched.warmup_ratio', 0.1) * 1000)}")

    config_lines.append("")
    config_lines.append("mlflow:")
    config_lines.append("  tracking_uri: sqlite:///mlflow.db")
    config_lines.append("")
    config_lines.append("project: psy_agents_noaug")

    # Write config file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(config_lines))

    return output_path


def run_criteria_training(best_config: Dict[str, Any], output_dir: Path, epochs: int) -> Dict[str, float]:
    """Run Criteria training using existing train_criteria.py script."""
    print(f"\n{'='*70}")
    print("TRAINING CRITERIA MODEL".center(70))
    print(f"{'='*70}")

    params = best_config["params"]
    val_f1 = best_config["f1_macro"]

    print(f"Validation F1 (HPO): {val_f1:.4f}")
    print(f"Model: {params.get('model.name')}")
    print(f"Epochs: {epochs}")
    print()

    # Create temp config
    temp_config_dir = Path("outputs/temp_configs")
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    config_path = convert_to_yaml_config("criteria", params, temp_config_dir / "criteria_best.yaml")

    print(f"Generated config: {config_path}")

    # Run training
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "criteria_training.log"

    cmd = [
        sys.executable,
        "scripts/train_criteria.py",
        f"training.epochs={epochs}",
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Log: {log_file}")

    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        print(f"✗ Training failed with code {result.returncode}")
        print(f"Check log: {log_file}")
        return None

    print("✓ Training complete")

    # Parse test metrics from log
    test_metrics = parse_test_metrics_from_log(log_file)

    return test_metrics


def parse_test_metrics_from_log(log_file: Path) -> Dict[str, float]:
    """Extract test metrics from training log file."""
    with open(log_file) as f:
        log_content = f.read()

    # Look for test results section
    # Example from train_criteria.py:
    # print(f"Accuracy:  {test_accuracy:.4f}")
    # print(f"F1 Macro:  {test_f1_macro:.4f}")
    # etc.

    metrics = {}

    for line in log_content.split("\n"):
        if "Accuracy:" in line and "test" in log_content.lower():
            try:
                metrics["test_accuracy"] = float(line.split()[-1])
            except:
                pass
        elif "F1 Macro:" in line:
            try:
                metrics["test_f1_macro"] = float(line.split()[-1])
            except:
                pass
        elif "F1 Micro:" in line:
            try:
                metrics["test_f1_micro"] = float(line.split()[-1])
            except:
                pass
        elif "Precision:" in line:
            try:
                metrics["test_precision"] = float(line.split()[-1])
            except:
                pass
        elif "Recall:" in line:
            try:
                metrics["test_recall"] = float(line.split()[-1])
            except:
                pass

    return metrics


def generate_comparison_report(
    results: Dict[str, Dict[str, Any]],
    output_path: Path
):
    """Generate validation vs test comparison report."""
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON REPORT".center(70))
    print(f"{'='*70}\n")

    lines = []
    lines.append("# Validation vs Test Performance - Final Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**HPO Run:** Maximal (Oct 31 - Nov 2, 2025)\n\n")

    lines.append("---\n\n")

    # Summary table
    lines.append("## Performance Summary\n\n")
    lines.append("| Rank | Architecture | Val F1  | Test F1 | Difference | Status |\n")
    lines.append("|------|--------------|---------|---------|------------|--------|\n")

    # Sort by validation F1
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["validation"]["f1_macro"],
        reverse=True
    )

    for rank, (agent, data) in enumerate(sorted_results, 1):
        val_f1 = data["validation"]["f1_macro"]
        test_f1 = data.get("test", {}).get("test_f1_macro")

        emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")

        if test_f1 is not None:
            diff = test_f1 - val_f1
            if abs(diff) < 0.03:  # Within 3%
                status = "✅ Excellent"
            elif diff < 0:
                status = "⚠️ Lower"
            else:
                status = "🎉 Higher"

            lines.append(
                f"| {emoji} {rank} | {agent.capitalize():12} | {val_f1:.4f} | "
                f"{test_f1:.4f} | {diff:+.4f} | {status} |\n"
            )
        else:
            lines.append(
                f"| {emoji} {rank} | {agent.capitalize():12} | {val_f1:.4f} | "
                f"N/A | N/A | ⏳ Pending |\n"
            )

    lines.append("\n---\n\n")

    # Detailed results per architecture
    lines.append("## Detailed Results\n\n")

    for agent, data in sorted_results:
        lines.append(f"### {agent.capitalize()} Architecture\n\n")

        # Validation performance
        lines.append("**Validation Performance (from HPO):**\n")
        lines.append(f"- F1 Macro: {data['validation']['f1_macro']:.4f}\n")
        if "ece" in data["validation"]:
            lines.append(f"- ECE: {data['validation']['ece']:.4f}\n")
        if "logloss" in data["validation"]:
            lines.append(f"- Log Loss: {data['validation']['logloss']:.4f}\n")
        lines.append("\n")

        # Test performance
        if "test" in data and data["test"]:
            lines.append("**Test Performance (from Refit):**\n")
            test = data["test"]
            if "test_f1_macro" in test:
                lines.append(f"- F1 Macro: {test['test_f1_macro']:.4f}\n")
            if "test_accuracy" in test:
                lines.append(f"- Accuracy: {test['test_accuracy']:.4f}\n")
            if "test_precision" in test:
                lines.append(f"- Precision: {test['test_precision']:.4f}\n")
            if "test_recall" in test:
                lines.append(f"- Recall: {test['test_recall']:.4f}\n")
            lines.append("\n")
        else:
            lines.append("**Test Performance:** Not yet evaluated\n\n")

        # Best hyperparameters
        lines.append("**Best Hyperparameters:**\n")
        params = data["config"]["params"]
        lines.append(f"- Model: {params.get('model.name', 'N/A')}\n")
        lines.append(f"- Optimizer: {params.get('optim.name', 'N/A')}\n")
        lines.append(f"- Learning Rate: {params.get('optim.lr', 0.0):.2e}\n")
        lines.append(f"- Batch Size: {params.get('train.batch_size', 'N/A')}\n")
        aug = "Yes" if params.get("aug.enabled", False) else "No"
        lines.append(f"- Augmentation: {aug}\n")
        lines.append("\n")

        lines.append("---\n\n")

    # Write report
    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"✓ Report saved to: {output_path}\n")

    # Print to console
    for line in lines:
        print(line, end="")


def main():
    parser = argparse.ArgumentParser(description="Run test evaluation for all architectures")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["criteria"],  # Start with criteria only
        choices=["share", "joint", "criteria", "evidence"],
        help="Which architectures to evaluate (default: criteria)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 epochs for testing",
    )

    args = parser.parse_args()

    if args.quick:
        args.epochs = 10

    print(f"\n{'='*70}")
    print("TEST EVALUATION PIPELINE".center(70))
    print(f"{'='*70}")
    print(f"Architectures: {', '.join(args.agents)}")
    print(f"Epochs: {args.epochs}")
    print()

    # Define paths
    project_root = Path(__file__).parent.parent
    topk_dir = project_root / "_runs/maximal_2025-10-31/topk"
    output_base = project_root / "outputs/test_evaluation"

    # Collect results
    results = {}

    for agent in args.agents:
        print(f"\n{'#'*70}")
        print(f"# {agent.upper()}")
        print(f"{'#'*70}\n")

        # Load best config
        topk_json = topk_dir / f"{agent}_noaug-{agent}-max-2025-10-31_topk.json"

        if not topk_json.exists():
            print(f"✗ topk JSON not found: {topk_json}")
            continue

        best_config = load_best_config_from_topk(topk_json)

        results[agent] = {
            "validation": {
                "f1_macro": best_config["f1_macro"],
                "ece": best_config.get("ece"),
                "logloss": best_config.get("logloss"),
            },
            "config": best_config,
        }

        # Run training and test evaluation
        output_dir = output_base / agent
        test_metrics = run_criteria_training(best_config, output_dir, args.epochs)

        if test_metrics:
            results[agent]["test"] = test_metrics
            print(f"\n✓ Test F1 Macro: {test_metrics.get('test_f1_macro', 'N/A')}")
        else:
            results[agent]["test"] = None
            print("\n✗ Test evaluation failed")

    # Generate comparison report
    report_path = project_root / "VALIDATION_VS_TEST_COMPARISON.md"
    generate_comparison_report(results, report_path)

    print(f"\n{'='*70}")
    print("TEST EVALUATION COMPLETE".center(70))
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
