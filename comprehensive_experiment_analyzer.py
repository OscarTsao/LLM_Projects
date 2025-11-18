#!/usr/bin/env python3
"""
Comprehensive Experimental Results Analyzer
Scans all GPU project directories and extracts experimental results
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re
from datetime import datetime

class ExperimentAnalyzer:
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

        # Check data for hints
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

    def extract_metrics(self, data: dict) -> dict:
        """Extract all metrics from data"""
        metrics = {}

        if not isinstance(data, dict):
            return metrics

        # Common metric names
        metric_keys = [
            "accuracy", "acc",
            "micro_f1", "micro_f1_score", "f1_micro",
            "macro_f1", "macro_f1_score", "f1_macro",
            "weighted_f1", "weighted_f1_score", "f1_weighted",
            "precision", "recall", "f1", "f1_score",
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
            data.get("performance", {})
        ]

        for search_dict in search_dicts:
            if isinstance(search_dict, dict):
                for key in metric_keys:
                    if key in search_dict and key not in metrics:
                        value = search_dict[key]
                        if isinstance(value, (int, float)):
                            metrics[key] = float(value)

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
            data.get("test_metrics", {}).get("classification_report", {})
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

    def extract_experiment_from_file(self, file_path: Path) -> Optional[dict]:
        """Extract experiment information from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

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
        metrics = self.extract_metrics(data)
        training_config = self.extract_training_config(data)

        # Get timestamp
        timestamp = data.get("timestamp") or data.get("created_at")
        if not timestamp and file_path.exists():
            timestamp = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

        # Get data version
        data_version = data.get("data_version") or data.get("dataset_version") or "unknown"

        # Get split
        split = data.get("split", "unknown")
        if "test" in path_str.lower():
            split = "test"
        elif "val" in path_str.lower():
            split = "val"
        elif "train" in path_str.lower():
            split = "train"

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

        return experiment

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
            "**/best_params.json"
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
                        experiment = self.extract_experiment_from_file(file_path)
                        if experiment and experiment["metrics"]:
                            self.all_experiments.append(experiment)

        print(f"\nTotal experiments found: {len(self.all_experiments)}")
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
            if "evidence" in task_type or "span" in task_type:
                primary_metric = "f1_score"
                fallback_metrics = ["f1", "exact_match", "macro_f1"]
            elif "reranker" in task_type:
                primary_metric = "ndcg"
                fallback_metrics = ["ndcg@5", "mrr", "map"]
            else:  # classification tasks
                primary_metric = "macro_f1"
                fallback_metrics = ["micro_f1", "weighted_f1", "f1", "accuracy"]

            # Find best by primary metric
            best_exp = None
            best_value = -1

            for exp in experiments:
                metrics = exp["metrics"]

                # Try primary metric first
                value = metrics.get(primary_metric)
                if value is None:
                    # Try fallback metrics
                    for fallback in fallback_metrics:
                        value = metrics.get(fallback)
                        if value is not None:
                            break

                if value is not None and value > best_value:
                    best_value = value
                    best_exp = exp

            if best_exp:
                # Create a key for the best experiments dict
                config_key = f"{model_family}_{single_vs_multi}_{augmentation}"

                if task_type not in self.best_experiments:
                    self.best_experiments[task_type] = {}

                self.best_experiments[task_type][config_key] = {
                    "best_experiment_id": best_exp["experiment_id"],
                    "project_path": best_exp["project_path"],
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
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENTAL RESULTS ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Base Path: {self.base_path}\n\n")

            # Overall statistics
            f.write("=" * 80 + "\n")
            f.write("1. OVERALL STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Experiments: {len(self.all_experiments)}\n\n")

            # By task type
            task_counts = defaultdict(int)
            for exp in self.all_experiments:
                task_counts[exp["task_type"]] += 1

            f.write("Experiments by Task Type:\n")
            for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {task:40s}: {count:4d}\n")
            f.write("\n")

            # By model family
            model_counts = defaultdict(int)
            for exp in self.all_experiments:
                model_counts[exp["model_family"]] += 1

            f.write("Experiments by Model Family:\n")
            for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {model:40s}: {count:4d}\n")
            f.write("\n")

            # By GPU
            gpu_counts = defaultdict(int)
            for exp in self.all_experiments:
                gpu_counts[exp["gpu_type"]] += 1

            f.write("Experiments by GPU Type:\n")
            for gpu, count in sorted(gpu_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {gpu:40s}: {count:4d}\n")
            f.write("\n")

            # Performance leaderboards
            f.write("=" * 80 + "\n")
            f.write("2. PERFORMANCE LEADERBOARDS\n")
            f.write("=" * 80 + "\n\n")

            self.write_leaderboard(f, "criteria_matching", "macro_f1")
            self.write_leaderboard(f, "evidence_sentence", "f1_score")
            self.write_leaderboard(f, "multi_task_criteria_evidence", "macro_f1")

            # Key findings
            f.write("=" * 80 + "\n")
            f.write("3. KEY FINDINGS\n")
            f.write("=" * 80 + "\n\n")

            self.write_key_findings(f)

            # Recommendations
            f.write("=" * 80 + "\n")
            f.write("4. RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            self.write_recommendations(f)

        print(f"  Saved: {report_file}")

    def write_leaderboard(self, f, task_type: str, metric: str):
        """Write top 10 leaderboard for a task type"""
        # Filter experiments for this task type
        task_experiments = [
            exp for exp in self.all_experiments
            if exp["task_type"] == task_type and exp["split"] == "test"
        ]

        if not task_experiments:
            f.write(f"Top 10 {task_type} Experiments:\n")
            f.write("  No experiments found for this task type.\n\n")
            return

        # Sort by metric (try multiple metric names)
        metric_variants = [metric, metric.replace("_", ""), "f1", "macro_f1", "accuracy"]

        scored_experiments = []
        for exp in task_experiments:
            for m in metric_variants:
                if m in exp["metrics"]:
                    scored_experiments.append((exp, exp["metrics"][m]))
                    break

        scored_experiments.sort(key=lambda x: x[1], reverse=True)

        f.write(f"Top 10 {task_type} Experiments (by {metric}):\n")
        f.write("-" * 80 + "\n")

        for i, (exp, score) in enumerate(scored_experiments[:10], 1):
            f.write(f"{i:2d}. {score:.4f} | {exp['project_path']:50s} | {exp['experiment_id']:15s}\n")
            f.write(f"     Model: {exp['model_name']:40s} Aug: {exp['augmentation']:15s}\n")
        f.write("\n")

    def write_key_findings(self, f):
        """Write key findings section"""
        f.write("Data Augmentation Impact:\n")
        f.write("-" * 80 + "\n")

        # Compare augmented vs non-augmented for each task
        for task_type in ["criteria_matching", "evidence_sentence", "multi_task_criteria_evidence"]:
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
                # Get average performance
                aug_scores = []
                no_aug_scores = []

                for exp in aug_experiments:
                    score = exp["metrics"].get("macro_f1") or exp["metrics"].get("f1") or exp["metrics"].get("accuracy")
                    if score:
                        aug_scores.append(score)

                for exp in no_aug_experiments:
                    score = exp["metrics"].get("macro_f1") or exp["metrics"].get("f1") or exp["metrics"].get("accuracy")
                    if score:
                        no_aug_scores.append(score)

                if aug_scores and no_aug_scores:
                    avg_aug = sum(aug_scores) / len(aug_scores)
                    avg_no_aug = sum(no_aug_scores) / len(no_aug_scores)
                    improvement = ((avg_aug - avg_no_aug) / avg_no_aug) * 100

                    f.write(f"\n{task_type}:\n")
                    f.write(f"  Augmented (n={len(aug_scores)}): {avg_aug:.4f}\n")
                    f.write(f"  No Aug (n={len(no_aug_scores)}): {avg_no_aug:.4f}\n")
                    f.write(f"  Improvement: {improvement:+.2f}%\n")

        f.write("\n")

    def write_recommendations(self, f):
        """Write recommendations section"""
        f.write("Based on the experimental results:\n\n")

        f.write("1. Best Performing Models:\n")
        # Find top models across all tasks
        model_scores = defaultdict(list)
        for exp in self.all_experiments:
            if exp["split"] == "test":
                score = exp["metrics"].get("macro_f1") or exp["metrics"].get("f1") or exp["metrics"].get("accuracy")
                if score:
                    model_scores[exp["model_family"]].append(score)

        model_avg = {model: sum(scores)/len(scores) for model, scores in model_scores.items() if scores}
        top_models = sorted(model_avg.items(), key=lambda x: x[1], reverse=True)[:3]

        for model, avg_score in top_models:
            f.write(f"   - {model}: Average score {avg_score:.4f}\n")
        f.write("\n")

        f.write("2. Best Augmentation Methods:\n")
        aug_scores = defaultdict(list)
        for exp in self.all_experiments:
            if exp["split"] == "test" and exp["augmentation"] != "unknown":
                score = exp["metrics"].get("macro_f1") or exp["metrics"].get("f1") or exp["metrics"].get("accuracy")
                if score:
                    aug_scores[exp["augmentation"]].append(score)

        aug_avg = {aug: sum(scores)/len(scores) for aug, scores in aug_scores.items() if scores}
        top_augs = sorted(aug_avg.items(), key=lambda x: x[1], reverse=True)[:3]

        for aug, avg_score in top_augs:
            f.write(f"   - {aug}: Average score {avg_score:.4f}\n")
        f.write("\n")

        f.write("3. Recommended Next Steps:\n")
        f.write("   - Focus on the best performing model families\n")
        f.write("   - Continue with effective data augmentation methods\n")
        f.write("   - Consider ensemble approaches combining top models\n")
        f.write("   - Investigate why certain configurations underperform\n")
        f.write("\n")

def main():
    analyzer = ExperimentAnalyzer()

    # Scan all experiments
    analyzer.scan_all_experiments()

    # Find best experiments
    analyzer.find_best_experiments()

    # Save results
    analyzer.save_results()

    print("\nAnalysis complete!")
    print(f"Total experiments analyzed: {len(analyzer.all_experiments)}")
    print(f"Best configurations found: {sum(len(v) for v in analyzer.best_experiments.values())}")

if __name__ == "__main__":
    main()
