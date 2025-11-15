#!/usr/bin/env python3
"""Simple test evaluation using best WITH-AUG HPO configurations.

This script:
1. Loads best parameters from hpo_best_configs_summary.json
2. Trains models with those exact parameters
3. Evaluates on test set
4. Reports test performance

Usage:
    python scripts/simple_test_eval.py --arch criteria --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psy_agents_noaug.hpo.evaluation import run_experiment


def map_indices_to_names(params):
    """Map parameter indices to actual names."""
    # Model mapping
    model_map = {
        0: "distilbert-base-uncased",
        1: "bert-base-uncased",
        2: "roberta-base",
        3: "albert-base-v2",
        4: "google/electra-base-discriminator",
        5: "microsoft/deberta-v3-base",
        6: "YituTech/conv-bert-base",
        7: "google/mobilebert-uncased",
    }

    # Optimizer mapping
    opt_map = {
        0: "adamw",
        1: "adam",
        2: "adafactor",
        3: "adamw_8bit",
        4: "lion",
        5: "lamb",
    }

    # Scheduler mapping
    sched_map = {
        0: "linear",
        1: "cosine",
        2: "cosine_restart",
        3: "polynomial",
        4: "constant",
    }

    # Null strategy mapping (must match null_policy.py supported strategies)
    null_strategy_map = {
        0: "none",
        1: "threshold",
        2: "ratio",
        3: "calibrated",
    }

    # Pooling mapping
    pooling_map = {
        0: "cls",
        1: "mean",
        2: "max",
    }

    # Activation mapping
    activation_map = {
        0: "gelu",
        1: "relu",
        2: "swish",
        3: "tanh",
    }

    # Method strategy mapping
    method_strategy_map = {
        0: "all",
        1: "contextual",
        2: "random",
    }

    # Convert indices
    if "model.name" in params:
        val = params["model.name"]
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace(".", "", 1).isdigit()):
            idx = int(float(val))
            params["model.name"] = model_map.get(idx, "distilbert-base-uncased")
        else:
            params["model.name"] = str(val)

    if "optim.name" in params:
        val = params["optim.name"]
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace(".", "", 1).isdigit()):
            idx = int(float(val))
            params["optim.name"] = opt_map.get(idx, "adamw")
        else:
            params["optim.name"] = str(val)

    if "sched.name" in params:
        val = params["sched.name"]
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace(".", "", 1).isdigit()):
            idx = int(float(val))
            params["sched.name"] = sched_map.get(idx, "linear")
        else:
            params["sched.name"] = str(val)

    if "null.strategy" in params:
        val = params["null.strategy"]
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace(".", "", 1).isdigit()):
            idx = int(float(val))
            params["null.strategy"] = null_strategy_map.get(idx, "none")
        else:
            params["null.strategy"] = str(val)

    if "head.pooling" in params:
        val = params["head.pooling"]
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace(".", "", 1).isdigit()):
            idx = int(float(val))
            params["head.pooling"] = pooling_map.get(idx, "cls")
        else:
            params["head.pooling"] = str(val)

    if "head.activation" in params:
        val = params["head.activation"]
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace(".", "", 1).isdigit()):
            idx = int(float(val))
            params["head.activation"] = activation_map.get(idx, "gelu")
        else:
            params["head.activation"] = str(val)

    if "aug.method_strategy" in params:
        val = params["aug.method_strategy"]
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace(".", "", 1).isdigit()):
            idx = int(float(val))
            params["aug.method_strategy"] = method_strategy_map.get(idx, "all")
        else:
            params["aug.method_strategy"] = str(val)

    # Convert antonym_guard boolean
    if "aug.antonym_guard" in params:
        val = float(params["aug.antonym_guard"])
        params["aug.antonym_guard"] = "on" if val > 0.5 else "off"

    # Fix gradient accumulation (cannot be 0)
    if "train.grad_accum" in params:
        val = int(float(params["train.grad_accum"]))
        params["train.grad_accum"] = max(1, val)  # Minimum 1

    # Fix batch size (cannot be 0)
    if "train.batch_size" in params:
        val = int(float(params["train.batch_size"]))
        params["train.batch_size"] = max(1, val)  # Minimum 1

    # Disable gradient checkpointing when using gradient accumulation > 1
    # to avoid double backward errors
    if "model.gradient_checkpointing" in params and "train.grad_accum" in params:
        grad_accum = int(float(params["train.grad_accum"]))
        if grad_accum > 1:
            params["model.gradient_checkpointing"] = 0

    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["criteria", "evidence", "share", "joint"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--config", default="hpo_best_configs_summary.json")
    parser.add_argument("--model", default=None, help="Override pretrained model name (HF identifier)")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable augmentation regardless of config",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience for run_experiment",
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"TEST EVALUATION: {args.arch.upper()}".center(80))
    print(f"{'='*80}\n")

    # Load best config
    with open(args.config) as f:
        configs = json.load(f)

    if args.arch not in configs:
        print(f"✗ No config found for {args.arch}")
        return 1

    config = configs[args.arch]
    params = config["params"]

    print(f"Best Trial: #{config['best_trial']}")
    print(f"Validation F1: {config['best_value']:.4f}")
    print(f"Training Epochs: {args.epochs}\n")

    # Convert string params to correct types
    for key, value in params.items():
        try:
            params[key] = json.loads(value)
        except:
            pass

    # Manual overrides
    if args.model:
        params["model.name"] = args.model

    if args.no_aug:
        params["aug.enabled"] = 0
        params["aug.p_apply"] = 0.0
        params["aug.ops_per_sample"] = 0
        params["aug.max_replace"] = 0.0

    # Map indices to names
    params = map_indices_to_names(params)
    params["agent"] = args.arch

    # Print key params
    print("Configuration:")
    print(f"  Model: {params.get('model.name')}")
    print(f"  Optimizer: {params.get('optim.name')}")
    print(f"  LR: {params.get('optim.lr'):.2e}")
    print(f"  Batch Size: {params.get('train.batch_size')}")
    print(f"  Augmentation: {'Yes' if params.get('aug.enabled') else 'No'}")
    print()

    # Run evaluation
    print("Starting training and validation evaluation...")

    try:
        result = run_experiment(
            agent=args.arch,
            params=params,
            epochs=args.epochs,
            seeds=[42],
            patience=args.patience,
            max_samples=None,
        )

        print(f"\n✓ Evaluation complete!")
        print(f"  F1 Macro: {result.get('f1_macro', 0):.4f}")
        print(f"  ECE: {result.get('ece', 0):.4f}")
        print(f"  Log Loss: {result.get('logloss', 0):.4f}")
        print(f"  Runtime: {result.get('runtime_s', 0):.1f}s")

        # Save results
        output_dir = Path(f"outputs/test_eval_{args.arch}")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "results.json", "w") as f:
            json.dump({
                "validation_f1_hpo": config['best_value'],
                "validation_f1_refit": result.get('f1_macro'),
                **result
            }, f, indent=2)

        print(f"\n✓ Results saved to: {output_dir}/results.json")

        return 0

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
