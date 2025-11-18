#!/usr/bin/env python3
"""
Enhanced Comprehensive Experimental Results Analyzer
Handles multiple file formats and extracts all experimental results
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re
from datetime import datetime

class EnhancedExperimentAnalyzer:
    def __init__(self, base_path: str = "/home/user/LLM_Projects"):
        self.base_path = Path(base_path)
        self.gpu_dirs = ["2080_LLM", "3090_LLM", "4070ti_LLM", "4090_LLM"]
        self.all_experiments = []
        self.best_experiments = defaultdict(dict)

    def extract_gpu_type(self, path: str) -> str:
        """Extract GPU type from path"""
        for gpu in self.gpu_dirs:
            if gpu in path:
                return gpu.replace("_LLM", "")
        return "unknown"

    def extract_project_name(self, path: str) -> str:
        """Extract project name from path"""
        parts = Path(path).parts
        for i, part in enumerate(parts):
            if part in self.gpu_dirs and i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"

    def infer_task_type(self, path: str, data: dict) -> str:
        """Infer task type from path and data"""
        path_lower = path.lower()

        # Check for multi-task indicators in data
        if isinstance(data, dict):
            # Check for multi-task metrics
            training_metrics = data.get("training_metrics", {})
            test_metrics = data.get("test_metrics", {})

            has_ev = any("_ev_" in k for k in {**training_metrics, **test_metrics}.keys())
            has_cri = any("_cri_" in k for k in {**training_metrics, **test_metrics}.keys())

            if has_ev and has_cri:
                return "multi_task_criteria_evidence"

        # Check path for hints
        if "criteria" in path_lower and "evidence" in path_lower:
            if "multi" in path_lower or "both" in path_lower:
                return "multi_task_criteria_evidence"

        if "criteria" in path_lower:
            return "criteria_matching"
        if "evidence" in path_lower:
            if "span" in path_lower:
                return "evidence_span"
            return "evidence_sentence"
        if "reranker" in path_lower:
            return "reranker"
        if "multi" in path_lower:
            return "multi_task"

        # Check data for task type field
        if isinstance(data, dict):
            task = data.get("task_type") or data.get("task") or data.get("config", {}).get("task_type")
            if task:
                return task

        return "unknown"

    def infer_model_info(self, path: str, data: dict) -> tuple:
        """Infer model family and name"""
        model_name = None

        # Try to get from data
        if isinstance(data, dict):
            model_name = (
                data.get("model_name") or
                data.get("model") or
                data.get("config", {}).get("model_name") or
                data.get("config", {}).get("model") or
                data.get("model_config", {}).get("name")
            )

        # Try to infer from path
        if not model_name:
            path_lower = path.lower()
            if "deberta" in path_lower:
                if "v3" in path_lower:
                    model_name = "microsoft/deberta-v3-base"
                else:
                    model_name = "microsoft/deberta-base"
            elif "roberta" in path_lower:
                model_name = "roberta-base"
            elif "bert" in path_lower:
                model_name = "bert-base-uncased"
            elif "llama" in path_lower:
                model_name = "meta-llama/Llama-2-7b"
            elif "qwen" in path_lower:
                model_name = "Qwen/Qwen-7B"
            elif "gemma" in path_lower:
                model_name = "google/gemma-7b"

        # Determine model family
        if model_name:
            model_name_lower = model_name.lower()
            if "deberta-v3" in model_name_lower:
                family = "deberta_v3"
            elif "deberta" in model_name_lower:
                family = "deberta"
            elif "roberta" in model_name_lower:
                family = "roberta"
            elif "bert" in model_name_lower:
                family = "bert"
            elif "llama" in model_name_lower:
                family = "llama"
            elif "qwen" in model_name_lower:
                family = "qwen"
            elif "gemma" in model_name_lower:
                family = "gemma"
            else:
                family = "unknown"
        else:
            family = "unknown"

        return family, model_name

    def infer_augmentation(self, path: str, data: dict) -> str:
        """Infer data augmentation method"""
        # Check data first
        if isinstance(data, dict):
            aug = (
                data.get("augmentation") or
                data.get("data_augmentation") or
                data.get("config", {}).get("augmentation")
            )
            if aug:
                return aug

        # Check path
        path_lower = path.lower()
        if "noaug" in path_lower or "no_aug" in path_lower:
            return "none"
        if "nlpaug" in path_lower:
            return "nlpaug"
        if "eda" in path_lower:
            return "eda"
        if "backtrans" in path_lower:
            return "backtranslation"
        if "paraphrase" in path_lower:
            return "llm_paraphrase"
        if "textattack" in path_lower:
            return "textattack"
        if "hybrid" in path_lower:
            return "hybrid"
        if "dataaug" in path_lower or "data_aug" in path_lower:
            return "mixed"

        return "unknown"

    def extract_metrics_from_nested_structure(self, data: dict, prefix: str = "") -> dict:
        """Extract metrics from nested structures like training_metrics/test_metrics"""
        metrics = {}

        if not isinstance(data, dict):
            return metrics

        # For multi-task models with _ev_ and _cri_ metrics
        has_ev = any("_ev_" in k for k in data.keys())
        has_cri = any("_cri_" in k for k in data.keys())

        if has_ev or has_cri:
            # Extract evidence metrics
            for key in ["macro_f1", "accuracy", "precision", "recall", "f1"]:
                ev_key = f"{prefix}_ev_{key}" if prefix else f"_ev_{key}"
                for data_key in data.keys():
                    if ev_key in data_key:
                        metrics[f"evidence_{key}"] = data[data_key]

            # Extract criteria metrics
            for key in ["macro_f1", "accuracy", "precision", "recall", "f1"]:
                cri_key = f"{prefix}_cri_{key}" if prefix else f"_cri_{key}"
                for data_key in data.keys():
                    if cri_key in data_key:
                        metrics[f"criteria_{key}"] = data[data_key]

            # Extract mean metrics
            for key in ["macro_f1_mean", "accuracy_mean"]:
                full_key = f"{prefix}_{key}" if prefix else key
                if full_key in data:
                    metrics[key] = data[full_key]

        return metrics

    def extract_metrics(self, data: dict) -> dict:
        """Extract all metrics from data with support for multiple formats"""
        metrics = {}

        if not isinstance(data, dict):
            return metrics

        # Format 1: Direct metrics at top level or in metrics/test_metrics dict
        metric_keys = [
            "accuracy", "acc",
            "micro_f1", "micro_f1_score", "f1_micro",
            "macro_f1", "macro_f1_score", "f1_macro",
            "weighted_f1", "weighted_f1_score", "f1_weighted",
            "precision", "precision_macro", "recall", "recall_macro",
            "f1", "f1_score",
            "roc_auc", "auc", "pr_auc",
            "exact_match", "em",
            "span_precision", "span_recall", "span_f1",
            "iou",
            "ndcg", "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
            "mrr", "map", "loss"
        ]

        # Search in common locations
        search_dicts = [
            data,
            data.get("metrics", {}),
            data.get("test_metrics", {}),
            data.get("eval_metrics", {}),
            data.get("results", {}),
            data.get("performance", {}),
            data.get("test", {}),
            data.get("val", {})
        ]

        for search_dict in search_dicts:
            if isinstance(search_dict, dict):
                for key in metric_keys:
                    if key in search_dict and key not in metrics:
                        value = search_dict[key]
                        if isinstance(value, (int, float)):
                            metrics[key] = float(value)

        # Format 2: Multi-task with training_metrics/test_metrics
        if "training_metrics" in data:
            training_metrics = self.extract_metrics_from_nested_structure(
                data["training_metrics"], "val"
            )
            for key, value in training_metrics.items():
                metrics[f"val_{key}"] = value

        if "test_metrics" in data:
            test_metrics = self.extract_metrics_from_nested_structure(
                data["test_metrics"], "test"
            )
            for key, value in test_metrics.items():
                metrics[f"test_{key}"] = value

        # Extract per-label metrics
        per_label = self.extract_per_label_metrics(data)
        if per_label:
            metrics["per_label"] = per_label

        return metrics

    def extract_per_label_metrics(self, data: dict) -> dict:
        """Extract per-label classification metrics"""
        per_label = {}

        # Important labels to track
        important_labels = [
            "SUICIDAL_THOUGHTS", "SELF_HARM", "HARM_TO_OTHERS",
            "DEPRESSED_MOOD", "ANHEDONIA", "SLEEP_DISTURBANCE",
            "APPETITE_CHANGE", "FATIGUE", "WORTHLESSNESS_GUILT",
            "CONCENTRATION_DIFFICULTY", "PSYCHOMOTOR_CHANGE"
        ]

        # Search for classification report
        report_locations = [
            data.get("classification_report", {}),
            data.get("per_label_metrics", {}),
            data.get("metrics", {}).get("classification_report", {}),
            data.get("test_metrics", {}).get("classification_report", {}),
            data.get("test", {}).get("classification_report", {})
        ]

        for report in report_locations:
            if isinstance(report, dict):
                for label in important_labels:
                    if label in report and isinstance(report[label], dict):
                        per_label[label] = {
                            "precision": report[label].get("precision"),
                            "recall": report[label].get("recall"),
                            "f1": report[label].get("f1-score") or report[label].get("f1")
                        }

        return per_label

    def extract_training_config(self, data: dict) -> dict:
        """Extract training configuration"""
        config = {}

        if not isinstance(data, dict):
            return config

        # Search in common locations
        search_dicts = [
            data,
            data.get("config", {}),
            data.get("training_config", {}),
            data.get("hyperparameters", {}),
            data.get("params", {})
        ]

        config_keys = {
            "batch_size": ["batch_size", "per_device_train_batch_size", "train_batch_size"],
            "learning_rate": ["learning_rate", "lr"],
            "num_epochs": ["num_epochs", "epochs", "num_train_epochs"],
            "optimizer": ["optimizer", "optimizer_type"],
            "max_length": ["max_length", "max_seq_length", "max_sequence_length"],
            "warmup_steps": ["warmup_steps"],
            "weight_decay": ["weight_decay"]
        }

        for search_dict in search_dicts:
            if isinstance(search_dict, dict):
                for config_name, possible_keys in config_keys.items():
                    if config_name not in config:
                        for key in possible_keys:
                            if key in search_dict:
                                config[config_name] = search_dict[key]
                                break

        return config

    def determine_split(self, file_path: Path, data: dict) -> List[str]:
        """Determine which splits are available in this file"""
        path_str = str(file_path).lower()
        splits = []

        # Check explicit test/val in path
        if "test" in path_str and "test" not in str(file_path.parent.name).lower():
            splits.append("test")
        elif "val" in path_str and "val" not in str(file_path.parent.name).lower():
            splits.append("val")

        # Check data structure
        if isinstance(data, dict):
            # Check for split field
            if "split" in data:
                splits.append(data["split"])

            # Check for nested test/val dicts
            if "test" in data and isinstance(data["test"], dict):
                if "test" not in splits:
                    splits.append("test")
            if "val" in data and isinstance(data["val"], dict):
                if "val" not in splits:
                    splits.append("val")

            # Check for test_metrics/training_metrics (training_metrics usually = val)
            if "test_metrics" in data:
                if "test" not in splits:
                    splits.append("test")
            if "training_metrics" in data:
                if "val" not in splits:
                    splits.append("val")

        # Default
        if not splits:
            if "test" in path_str:
                splits.append("test")
            elif "val" in path_str:
                splits.append("val")
            else:
                splits.append("unknown")

        return splits

    def extract_experiment_from_file(self, file_path: Path) -> List[dict]:
        """Extract experiment information from a single file - may return multiple experiments"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        path_str = str(file_path)

        # Extract experiment ID from path
        exp_id = None
        if "trial_" in path_str:
            match = re.search(r'trial_(\w+)', path_str)
            if match:
                exp_id = f"trial_{match.group(1)}"
        elif "fold_" in path_str:
            match = re.search(r'fold_(\d+)', path_str)
            if match:
                exp_id = f"fold_{match.group(1)}"
        else:
            exp_id = file_path.stem

        # Extract basic info
        gpu_type = self.extract_gpu_type(path_str)
        project_name = self.extract_project_name(path_str)
        project_path = f"{gpu_type}_LLM/{project_name}"

        # Extract task and model info
        task_type = self.infer_task_type(path_str, data)
        model_family, model_name = self.infer_model_info(path_str, data)
        augmentation = self.infer_augmentation(path_str, data)

        # Determine if single or multi-task
        single_vs_multi = "multi" if "multi" in task_type.lower() else "single"

        # Extract metrics and config
        all_metrics = self.extract_metrics(data)
        training_config = self.extract_training_config(data)

        # Get timestamp
        timestamp = data.get("timestamp") or data.get("created_at")
        if not timestamp and file_path.exists():
            timestamp = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

        # Get data version
        data_version = data.get("data_version") or data.get("dataset_version") or "unknown"

        # Determine splits available in this file
        splits = self.determine_split(file_path, data)

        # Create experiment entries for each split
        experiments = []

        for split in splits:
            # Filter metrics for this split
            if split == "test":
                # Get test metrics
                if "test" in data and isinstance(data["test"], dict):
                    metrics = self.extract_metrics({"metrics": data["test"]})
                elif "test_metrics" in data:
                    metrics = {}
                    for k, v in all_metrics.items():
                        if k.startswith("test_"):
                            metrics[k.replace("test_", "")] = v
                else:
                    # For files named test_metrics.json, use all metrics
                    if "test" in str(file_path).lower():
                        metrics = all_metrics
                    else:
                        continue

            elif split == "val":
                # Get val metrics
                if "val" in data and isinstance(data["val"], dict):
                    metrics = self.extract_metrics({"metrics": data["val"]})
                elif "training_metrics" in data:
                    metrics = {}
                    for k, v in all_metrics.items():
                        if k.startswith("val_"):
                            metrics[k.replace("val_", "")] = v
                else:
                    # For files named val_metrics.json, use all metrics
                    if "val" in str(file_path).lower():
                        metrics = all_metrics
                    else:
                        continue

            else:
                metrics = all_metrics

            # Skip if no meaningful metrics
            if not metrics or (len(metrics) == 1 and "per_label" in metrics):
                continue

            experiment = {
                "project_path": project_path,
                "experiment_id": exp_id,
                "result_file": path_str,
                "timestamp": timestamp,
                "task_type": task_type,
                "model_family": model_family,
                "model_name": model_name or "unknown",
                "single_vs_multi_task": single_vs_multi,
                "data_version": data_version,
                "augmentation": augmentation,
                "split": split,
                "gpu_type": gpu_type,
                **training_config,
                "metrics": metrics
            }

            # Add training time if available
            train_time = data.get("training_time_seconds") or data.get("train_time")
            if train_time:
                experiment["train_time_minutes"] = float(train_time) / 60 if train_time > 100 else train_time

            # Add best checkpoint if available
            checkpoint = data.get("best_checkpoint") or data.get("checkpoint_path")
            if checkpoint:
                experiment["best_checkpoint"] = checkpoint

            # Add run_id if available
            run_id = data.get("run_id")
            if run_id:
                experiment["run_id"] = run_id

            experiments.append(experiment)

        return experiments

    def scan_all_experiments(self):
        """Scan all GPU directories for experimental results"""
        print("Scanning for experimental results...")

        # File patterns to search for
        patterns = [
            "**/evaluation_report.json",
            "**/test_metrics.json",
            "**/val_metrics.json",
            "**/metrics.json",
            "**/test_results.json",
            "**/eval_results.json",
        ]

        files_found = set()

        for gpu_dir in self.gpu_dirs:
            gpu_path = self.base_path / gpu_dir
            if not gpu_path.exists():
                print(f"  Skipping {gpu_dir} (not found)")
                continue

            print(f"\n  Scanning {gpu_dir}...")

            for pattern in patterns:
                for file_path in gpu_path.glob(pattern):
                    if file_path not in files_found:
                        files_found.add(file_path)
                        experiments = self.extract_experiment_from_file(file_path)
                        self.all_experiments.extend(experiments)

        print(f"\nTotal experiments found: {len(self.all_experiments)}")
        print(f"Total files processed: {len(files_found)}")
        return self.all_experiments

    def find_best_experiments(self):
        """Find best experiments for each configuration"""
        print("\nFinding best experiments...")

        # Group experiments by (task_type, model_family, single_vs_multi, augmentation)
        groups = defaultdict(list)

        for exp in self.all_experiments:
            if exp["split"] != "test":  # Only consider test set results
                continue

            key = (
                exp["task_type"],
                exp["model_family"],
                exp["single_vs_multi_task"],
                exp["augmentation"]
            )
            groups[key].append(exp)

        # For each group, find the best experiment
        for key, experiments in groups.items():
            task_type, model_family, single_vs_multi, augmentation = key

            # Determine primary metric based on task type
            if "multi_task" in task_type:
                # For multi-task, prioritize mean metrics
                primary_metrics = ["macro_f1_mean", "accuracy_mean", "macro_f1", "f1"]
            elif "evidence" in task_type or "span" in task_type:
                primary_metrics = ["f1_score", "f1", "exact_match", "macro_f1"]
            elif "reranker" in task_type:
                primary_metrics = ["ndcg", "ndcg@5", "mrr", "map"]
            else:  # classification tasks
                primary_metrics = ["macro_f1", "f1_macro", "micro_f1", "weighted_f1", "f1", "accuracy"]

            # Find best by primary metric
            best_exp = None
            best_value = -1
            best_metric_name = None

            for exp in experiments:
                metrics = exp["metrics"]

                # Try each metric in priority order
                for metric_name in primary_metrics:
                    value = metrics.get(metric_name)
                    if value is not None and value > best_value:
                        best_value = value
                        best_exp = exp
                        best_metric_name = metric_name
                        break

            if best_exp and best_value > 0.05:  # Only include if performance is reasonable
                # Create a key for the best experiments dict
                config_key = f"{model_family}_{single_vs_multi}_{augmentation}"

                if task_type not in self.best_experiments:
                    self.best_experiments[task_type] = {}

                self.best_experiments[task_type][config_key] = {
                    "best_experiment_id": best_exp["experiment_id"],
                    "project_path": best_exp["project_path"],
                    "primary_metric": best_metric_name,
                    "primary_metric_value": best_value,
                    "metrics": best_exp["metrics"],
                    "model_name": best_exp["model_name"],
                    "config_summary": self.create_config_summary(best_exp)
                }

        return dict(self.best_experiments)

    def create_config_summary(self, exp: dict) -> str:
        """Create a summary string of configuration"""
        parts = []

        if exp.get("batch_size"):
            parts.append(f"bs={exp['batch_size']}")
        if exp.get("learning_rate"):
            parts.append(f"lr={exp['learning_rate']}")
        if exp.get("num_epochs"):
            parts.append(f"epochs={exp['num_epochs']}")
        if exp.get("augmentation") and exp["augmentation"] != "unknown":
            parts.append(f"aug={exp['augmentation']}")

        return ", ".join(parts) if parts else "N/A"

    def save_results(self):
        """Save all analysis results"""
        print("\nSaving results...")

        # Save all experiments
        all_exp_file = self.base_path / "all_experiments.json"
        with open(all_exp_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_experiments, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {all_exp_file}")

        # Save best experiments
        best_exp_file = self.base_path / "best_experiments_summary.json"
        with open(best_exp_file, 'w', encoding='utf-8') as f:
            json.dump(self.best_experiments, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {best_exp_file}")

        # Generate analysis report
        self.generate_analysis_report()

    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        report_file = self.base_path / "EXPERIMENT_ANALYSIS_REPORT.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(" " * 30 + "EXPERIMENTAL RESULTS ANALYSIS REPORT\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base Path: {self.base_path}\n\n")

            # Overall statistics
            f.write("=" * 100 + "\n")
            f.write("1. OVERALL STATISTICS\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Total Experiments: {len(self.all_experiments)}\n\n")

            # By task type
            task_counts = defaultdict(int)
            for exp in self.all_experiments:
                task_counts[exp["task_type"]] += 1

            f.write("Experiments by Task Type:\n")
            f.write("-" * 100 + "\n")
            for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {task:50s}: {count:4d}\n")
            f.write("\n")

            # By model family
            model_counts = defaultdict(int)
            for exp in self.all_experiments:
                model_counts[exp["model_family"]] += 1

            f.write("Experiments by Model Family:\n")
            f.write("-" * 100 + "\n")
            for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {model:50s}: {count:4d}\n")
            f.write("\n")

            # By GPU
            gpu_counts = defaultdict(int)
            for exp in self.all_experiments:
                gpu_counts[exp["gpu_type"]] += 1

            f.write("Experiments by GPU Type:\n")
            f.write("-" * 100 + "\n")
            for gpu, count in sorted(gpu_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {gpu:50s}: {count:4d}\n")
            f.write("\n")

            # By augmentation
            aug_counts = defaultdict(int)
            for exp in self.all_experiments:
                aug_counts[exp["augmentation"]] += 1

            f.write("Experiments by Data Augmentation:\n")
            f.write("-" * 100 + "\n")
            for aug, count in sorted(aug_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {aug:50s}: {count:4d}\n")
            f.write("\n")

            # Performance leaderboards
            f.write("=" * 100 + "\n")
            f.write("2. PERFORMANCE LEADERBOARDS\n")
            f.write("=" * 100 + "\n\n")

            self.write_leaderboard(f, "criteria_matching", ["macro_f1", "f1_macro", "accuracy"])
            self.write_leaderboard(f, "evidence_sentence", ["f1", "f1_score", "macro_f1"])
            self.write_leaderboard(f, "evidence_span", ["f1", "f1_score", "exact_match"])
            self.write_leaderboard(f, "multi_task_criteria_evidence", ["macro_f1_mean", "macro_f1"])
            self.write_leaderboard(f, "reranker", ["ndcg", "mrr"])

            # Key findings
            f.write("=" * 100 + "\n")
            f.write("3. KEY FINDINGS\n")
            f.write("=" * 100 + "\n\n")

            self.write_key_findings(f)

            # Best configurations summary
            f.write("=" * 100 + "\n")
            f.write("4. BEST CONFIGURATIONS BY TASK\n")
            f.write("=" * 100 + "\n\n")

            self.write_best_configs(f)

            # Recommendations
            f.write("=" * 100 + "\n")
            f.write("5. RECOMMENDATIONS\n")
            f.write("=" * 100 + "\n\n")

            self.write_recommendations(f)

        print(f"  Saved: {report_file}")

    def write_leaderboard(self, f, task_type: str, metric_names: List[str]):
        """Write top 10 leaderboard for a task type"""
        # Filter experiments for this task type
        task_experiments = [
            exp for exp in self.all_experiments
            if exp["task_type"] == task_type and exp["split"] == "test"
        ]

        if not task_experiments:
            f.write(f"\nTop 10 {task_type} Experiments:\n")
            f.write("  No experiments found for this task type.\n\n")
            return

        # Sort by metric
        scored_experiments = []
        for exp in task_experiments:
            for metric in metric_names:
                if metric in exp["metrics"]:
                    scored_experiments.append((exp, exp["metrics"][metric], metric))
                    break

        if not scored_experiments:
            f.write(f"\nTop 10 {task_type} Experiments:\n")
            f.write("  No valid metrics found.\n\n")
            return

        scored_experiments.sort(key=lambda x: x[1], reverse=True)

        f.write(f"\nTop 10 {task_type} Experiments (by {metric_names[0]}):\n")
        f.write("-" * 100 + "\n")

        for i, (exp, score, metric_used) in enumerate(scored_experiments[:10], 1):
            f.write(f"{i:2d}. {score:.4f} ({metric_used:15s}) | {exp['project_path']:40s} | {exp['experiment_id']:15s}\n")
            f.write(f"     Model: {exp['model_name']:40s} Aug: {exp['augmentation']:15s}\n")
        f.write("\n")

    def write_key_findings(self, f):
        """Write key findings section"""
        f.write("Data Augmentation Impact Analysis:\n")
        f.write("-" * 100 + "\n\n")

        # Compare augmented vs non-augmented for each task
        for task_type in set(exp["task_type"] for exp in self.all_experiments):
            if task_type == "unknown":
                continue

            aug_experiments = [
                exp for exp in self.all_experiments
                if exp["task_type"] == task_type and
                exp["split"] == "test" and
                exp["augmentation"] not in ["none", "unknown"]
            ]

            no_aug_experiments = [
                exp for exp in self.all_experiments
                if exp["task_type"] == task_type and
                exp["split"] == "test" and
                exp["augmentation"] == "none"
            ]

            if aug_experiments and no_aug_experiments:
                # Determine which metric to use
                if "multi_task" in task_type:
                    metric_keys = ["macro_f1_mean", "macro_f1", "f1", "accuracy"]
                else:
                    metric_keys = ["macro_f1", "f1_macro", "f1", "f1_score", "accuracy"]

                # Get scores
                aug_scores = []
                no_aug_scores = []

                for exp in aug_experiments:
                    for key in metric_keys:
                        if key in exp["metrics"]:
                            aug_scores.append(exp["metrics"][key])
                            break

                for exp in no_aug_experiments:
                    for key in metric_keys:
                        if key in exp["metrics"]:
                            no_aug_scores.append(exp["metrics"][key])
                            break

                if aug_scores and no_aug_scores:
                    avg_aug = sum(aug_scores) / len(aug_scores)
                    avg_no_aug = sum(no_aug_scores) / len(no_aug_scores)
                    improvement = ((avg_aug - avg_no_aug) / avg_no_aug) * 100 if avg_no_aug > 0 else 0

                    f.write(f"{task_type}:\n")
                    f.write(f"  With Augmentation (n={len(aug_scores):3d}): {avg_aug:.4f}\n")
                    f.write(f"  No Augmentation   (n={len(no_aug_scores):3d}): {avg_no_aug:.4f}\n")
                    f.write(f"  Improvement: {improvement:+.2f}%\n\n")

    def write_best_configs(self, f):
        """Write best configurations summary"""
        for task_type in sorted(self.best_experiments.keys()):
            f.write(f"\n{task_type.upper()}:\n")
            f.write("-" * 100 + "\n")

            configs = self.best_experiments[task_type]
            # Sort by primary metric value
            sorted_configs = sorted(
                configs.items(),
                key=lambda x: x[1].get("primary_metric_value", 0),
                reverse=True
            )

            for config_key, config_data in sorted_configs[:5]:  # Top 5 for each task
                f.write(f"\n  Configuration: {config_key}\n")
                f.write(f"    Experiment: {config_data.get('best_experiment_id', 'N/A')}\n")
                f.write(f"    Project: {config_data.get('project_path', 'N/A')}\n")
                f.write(f"    Model: {config_data.get('model_name', 'N/A')}\n")
                f.write(f"    Primary Metric: {config_data.get('primary_metric', 'N/A')} = {config_data.get('primary_metric_value', 0):.4f}\n")
                if config_data.get('config_summary') != "N/A":
                    f.write(f"    Config: {config_data['config_summary']}\n")
            f.write("\n")

    def write_recommendations(self, f):
        """Write recommendations section"""
        f.write("Based on the experimental results:\n\n")

        f.write("1. Top Performing Models:\n")
        f.write("-" * 100 + "\n")
        # Find top models across all tasks
        model_scores = defaultdict(list)
        for exp in self.all_experiments:
            if exp["split"] == "test" and exp["model_family"] != "unknown":
                # Get primary metric
                for key in ["macro_f1", "f1_macro", "f1", "f1_score", "accuracy"]:
                    if key in exp["metrics"]:
                        model_scores[exp["model_family"]].append(exp["metrics"][key])
                        break

        model_avg = {model: sum(scores)/len(scores) for model, scores in model_scores.items() if scores}
        top_models = sorted(model_avg.items(), key=lambda x: x[1], reverse=True)[:5]

        for model, avg_score in top_models:
            count = len(model_scores[model])
            f.write(f"   {model:20s}: Average score {avg_score:.4f} (n={count})\n")
        f.write("\n")

        f.write("2. Best Data Augmentation Methods:\n")
        f.write("-" * 100 + "\n")
        aug_scores = defaultdict(list)
        for exp in self.all_experiments:
            if exp["split"] == "test" and exp["augmentation"] not in ["unknown"]:
                for key in ["macro_f1", "f1_macro", "f1", "f1_score", "accuracy"]:
                    if key in exp["metrics"]:
                        aug_scores[exp["augmentation"]].append(exp["metrics"][key])
                        break

        aug_avg = {aug: sum(scores)/len(scores) for aug, scores in aug_scores.items() if scores}
        top_augs = sorted(aug_avg.items(), key=lambda x: x[1], reverse=True)[:5]

        for aug, avg_score in top_augs:
            count = len(aug_scores[aug])
            f.write(f"   {aug:20s}: Average score {avg_score:.4f} (n={count})\n")
        f.write("\n")

        f.write("3. GPU Performance Comparison:\n")
        f.write("-" * 100 + "\n")
        gpu_scores = defaultdict(list)
        for exp in self.all_experiments:
            if exp["split"] == "test" and exp["gpu_type"] != "unknown":
                for key in ["macro_f1", "f1_macro", "f1", "f1_score", "accuracy"]:
                    if key in exp["metrics"]:
                        gpu_scores[exp["gpu_type"]].append(exp["metrics"][key])
                        break

        gpu_avg = {gpu: sum(scores)/len(scores) for gpu, scores in gpu_scores.items() if scores}
        for gpu in sorted(gpu_avg.keys()):
            avg_score = gpu_avg[gpu]
            count = len(gpu_scores[gpu])
            f.write(f"   {gpu:20s}: Average score {avg_score:.4f} (n={count})\n")
        f.write("\n")

        f.write("4. Recommended Next Steps:\n")
        f.write("-" * 100 + "\n")
        f.write("   a. Focus development on the top-performing model families\n")
        f.write("   b. Continue using effective data augmentation methods that show >5% improvement\n")
        f.write("   c. For multi-task models, investigate configurations with macro_f1_mean > 0.7\n")
        f.write("   d. Consider ensemble approaches combining top-3 models from each task\n")
        f.write("   e. Retire or deprioritize configurations with consistent scores < 0.5\n")
        f.write("   f. Archive low-performing experiments to focus compute resources\n")
        f.write("\n")

def main():
    analyzer = EnhancedExperimentAnalyzer()

    # Scan all experiments
    analyzer.scan_all_experiments()

    # Find best experiments
    analyzer.find_best_experiments()

    # Save results
    analyzer.save_results()

    print("\nAnalysis complete!")
    print(f"Total experiments analyzed: {len(analyzer.all_experiments)}")
    print(f"Total task configurations: {sum(len(v) for v in analyzer.best_experiments.values())}")

if __name__ == "__main__":
    main()
