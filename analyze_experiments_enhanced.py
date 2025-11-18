#!/usr/bin/env python3
"""
增強版實驗結果分析腳本
全面分析 LLM_Projects 目錄中的所有實驗結果
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict


def extract_project_info_from_path(project_path: str) -> Dict[str, Any]:
    """從專案路徑推斷專案資訊"""
    project_name = Path(project_path).name
    parent_dir = Path(project_path).parent.name if Path(project_path).parent.name else ""

    # 推斷 GPU 類型
    gpu_match = re.search(r'(2080|3090|4070ti|4090)', parent_dir)
    gpu_type = gpu_match.group(1) if gpu_match else "unknown"

    # 推斷模型家族
    model_family = "unknown"
    if "DeBERTa" in project_name or "deberta" in project_name.lower():
        model_family = "deberta_v3"
    elif "Mental" in project_name or "mental" in project_name.lower():
        model_family = "mental_bert"
    elif "RoBERTa" in project_name or "roberta" in project_name.lower():
        model_family = "roberta"
    elif "BERT" in project_name or "bert" in project_name.lower():
        model_family = "bert"

    # 推斷資料增強類型
    augmentation = "unknown"
    if "NoAug" in project_name or "Baseline" in project_name:
        augmentation = "none"
    elif "DataAug" in project_name:
        augmentation = "hybrid"  # 預設為 hybrid，後續可能調整
    elif "EDA" in project_name:
        augmentation = "eda"
    elif "BackTrans" in project_name:
        augmentation = "backtranslation"

    # 推斷資料版本
    data_version = "redsm5_v2" if "redsm5" in project_name.lower() or "REDSM5" in project_name else "unknown"

    return {
        "gpu_type": gpu_type,
        "model_family": model_family,
        "augmentation": augmentation,
        "data_version": data_version
    }


def classify_task_and_mode(metrics: Dict) -> tuple[str, str]:
    """根據指標分類任務類型和單/多任務模式"""

    # 檢查是否為 multi-task (同時有 criteria 和 evidence)
    has_criteria = any(k.startswith(('criteria_', 'test_cri_', 'val_cri_')) for k in metrics.keys())
    has_evidence_sentence = any(k.startswith(('evidence_', 'test_ev_', 'val_ev_')) for k in metrics.keys())
    has_evidence_span = any('span' in k for k in metrics.keys())
    has_symptom = any('symptom' in k for k in metrics.keys())

    if has_criteria and (has_evidence_sentence or has_evidence_span):
        task_type = "multi_task_criteria_evidence"
        single_vs_multi = "multi"
    elif has_evidence_span:
        task_type = "evidence_span"
        single_vs_multi = "single"
    elif has_evidence_sentence:
        task_type = "evidence_sentence"
        single_vs_multi = "single"
    elif has_criteria:
        task_type = "criteria_matching"
        single_vs_multi = "single"
    elif has_symptom:
        task_type = "symptom_classification"
        single_vs_multi = "single"
    else:
        task_type = "unknown"
        single_vs_multi = "unknown"

    return task_type, single_vs_multi


def normalize_metrics(raw_metrics: Dict, file_type: str) -> Dict[str, Any]:
    """標準化指標格式"""
    normalized = {}

    # 處理不同來源的指標
    if file_type == "evaluation_report":
        # 從 evaluation_report.json 提取測試指標
        if "test_metrics" in raw_metrics:
            test_metrics = raw_metrics["test_metrics"]

            # Multi-task format
            if "test_cri_macro_f1" in test_metrics or "test_ev_macro_f1" in test_metrics:
                normalized["criteria_macro_f1"] = test_metrics.get("test_cri_macro_f1")
                normalized["criteria_accuracy"] = test_metrics.get("test_cri_accuracy")
                normalized["evidence_macro_f1"] = test_metrics.get("test_ev_macro_f1")
                normalized["evidence_accuracy"] = test_metrics.get("test_ev_accuracy")
                normalized["macro_f1_mean"] = test_metrics.get("test_macro_f1_mean")

            # Span extraction
            elif "evidence_binding" in test_metrics:
                eb = test_metrics["evidence_binding"]
                normalized["span_f1"] = eb.get("span_f1")
                normalized["precision"] = eb.get("precision")
                normalized["recall"] = eb.get("recall")
                normalized["exact_match"] = eb.get("exact_match")
                normalized["char_f1"] = eb.get("char_f1")

            # Single task metrics
            else:
                for key in ["macro_f1", "micro_f1", "accuracy", "roc_auc", "pr_auc", "precision", "recall"]:
                    if key in test_metrics:
                        normalized[key] = test_metrics[key]

    elif file_type == "test_metrics" or file_type == "val_metrics":
        # From test_metrics.json or val_metrics.json
        # Handle nested structure (val/test keys)
        if "test" in raw_metrics:
            metrics = raw_metrics["test"]
        elif "val" in raw_metrics:
            metrics = raw_metrics["val"]
        else:
            metrics = raw_metrics

        # Standard classification metrics
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "auc"]:
            if key in metrics:
                normalized[key] = metrics[key]

        # Macro metrics
        for key in ["precision_macro", "recall_macro", "f1_macro"]:
            if key in metrics:
                normalized[key] = metrics[key]

    elif file_type == "agent_metrics":
        # From Criteria_Evidence_Agent/metrics.json
        if "test_metrics" in raw_metrics:
            test_m = raw_metrics["test_metrics"]
            normalized["micro_f1"] = test_m.get("val_symptom_labels_micro_f1")
            normalized["macro_f1"] = test_m.get("val_symptom_labels_macro_f1")
            normalized["roc_auc"] = test_m.get("val_symptom_labels_roc_auc")
            normalized["loss"] = test_m.get("val_loss")

        if "best_metric" in raw_metrics:
            normalized["best_metric"] = raw_metrics["best_metric"]
        if "best_epoch" in raw_metrics:
            normalized["best_epoch"] = raw_metrics["best_epoch"]

    # Remove None values
    normalized = {k: v for k, v in normalized.items() if v is not None}

    return normalized


def process_experiment_file(file_path: Path, base_path: Path, file_type: str) -> Optional[Dict]:
    """處理單個實驗檔案"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 獲取相對路徑
        rel_path = file_path.relative_to(base_path)
        parts = rel_path.parts

        # 提取專案路徑
        if len(parts) >= 2:
            project_path = str(Path(parts[0]) / parts[1])
        else:
            project_path = str(parts[0])

        # 提取實驗 ID
        if file_type == "evaluation_report":
            # experiments/trial_xxx/evaluation_report.json
            experiment_id = parts[3] if len(parts) > 3 and parts[2] == "experiments" else "unknown"
        elif file_type in ["test_metrics", "val_metrics"]:
            # mlruns/1/xxx/artifacts/test_metrics.json OR outputs/.../test_metrics.json
            if "mlruns" in parts:
                experiment_id = f"mlrun_{parts[parts.index('mlruns') + 2]}"
            elif "outputs" in parts:
                # Find the most specific identifier
                idx = parts.index("outputs")
                if len(parts) > idx + 1:
                    experiment_id = f"output_{parts[idx + 1]}"
                else:
                    experiment_id = "output_unknown"
            else:
                experiment_id = "unknown"
        elif file_type == "agent_metrics":
            experiment_id = "agent_run"
        else:
            experiment_id = "unknown"

        # 提取專案資訊
        full_project_path = base_path / project_path
        project_info = extract_project_info_from_path(str(full_project_path))

        # 標準化指標
        normalized_metrics = normalize_metrics(data, file_type)

        # 分類任務類型
        task_type, single_vs_multi = classify_task_and_mode(normalized_metrics)

        # 如果從項目名稱可以更準確判斷，則覆蓋
        project_name = Path(project_path).name
        if "Criteria" in project_name and "Evidence" in project_name and task_type == "unknown":
            task_type = "multi_task_criteria_evidence"
            single_vs_multi = "multi"

        # 組合結果
        result = {
            "project_path": project_path,
            "experiment_id": experiment_id,
            "task_type": task_type,
            "model_family": project_info["model_family"],
            "single_vs_multi_task": single_vs_multi,
            "data_version": project_info["data_version"],
            "augmentation": project_info["augmentation"],
            "metrics": normalized_metrics,
            "per_label_metrics": {},  # TODO: 從詳細報告中提取
            "training_info": {},
            "file_path": str(file_path),
            "source_type": file_type
        }

        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def find_best_experiments(all_experiments: List[Dict]) -> List[Dict]:
    """找出每個組合的最佳實驗"""
    groups = defaultdict(list)

    for exp in all_experiments:
        if not exp["metrics"]:
            continue

        group_key = f"{exp['task_type']}__{exp['model_family']}__{exp['single_vs_multi_task']}__{exp['augmentation']}"
        groups[group_key].append(exp)

    best_experiments = []

    for group_key, experiments in groups.items():
        if not experiments:
            continue

        # 確定排名標準
        task_type = experiments[0]["task_type"]

        # 選擇主要指標
        if task_type == "evidence_span":
            ranking_key = "span_f1"
        elif task_type == "multi_task_criteria_evidence":
            ranking_key = "macro_f1_mean"
        elif "macro_f1" in experiments[0]["metrics"]:
            ranking_key = "macro_f1"
        elif "f1_macro" in experiments[0]["metrics"]:
            ranking_key = "f1_macro"
        elif "f1" in experiments[0]["metrics"]:
            ranking_key = "f1"
        elif "micro_f1" in experiments[0]["metrics"]:
            ranking_key = "micro_f1"
        elif "accuracy" in experiments[0]["metrics"]:
            ranking_key = "accuracy"
        else:
            ranking_key = None

        if not ranking_key:
            continue

        # 找出最佳實驗
        best_exp = None
        best_score = -1

        for exp in experiments:
            score = exp["metrics"].get(ranking_key, -1)
            if score is not None and score > best_score:
                best_score = score
                best_exp = exp

        if best_exp:
            parts = group_key.split("__")
            best_experiments.append({
                "group_key": group_key,
                "task_type": parts[0],
                "model_family": parts[1],
                "single_vs_multi_task": parts[2],
                "augmentation": parts[3],
                "best_experiment_id": best_exp["experiment_id"],
                "project_path": best_exp["project_path"],
                "best_metrics": best_exp["metrics"],
                "ranking_criterion": ranking_key,
                "num_experiments_in_group": len(experiments)
            })

    return best_experiments


def main():
    """主函數"""
    base_path = Path("/home/user/LLM_Projects")
    all_experiments = []

    print("開始全面掃描實驗檔案...")

    # 1. 處理 evaluation_report.json 檔案
    print("\n[1/4] 處理 evaluation_report.json 檔案...")
    eval_reports = list(base_path.glob("**/evaluation_report.json"))
    print(f"找到 {len(eval_reports)} 個檔案")

    for i, file_path in enumerate(eval_reports):
        if i % 50 == 0:
            print(f"  進度: {i}/{len(eval_reports)}")
        result = process_experiment_file(file_path, base_path, "evaluation_report")
        if result:
            all_experiments.append(result)

    # 2. 處理 test_metrics.json 檔案
    print("\n[2/4] 處理 test_metrics.json 檔案...")
    test_metrics = list(base_path.glob("**/test_metrics.json"))
    print(f"找到 {len(test_metrics)} 個檔案")

    for file_path in test_metrics:
        result = process_experiment_file(file_path, base_path, "test_metrics")
        if result:
            all_experiments.append(result)

    # 3. 處理 val_metrics.json 檔案
    print("\n[3/4] 處理 val_metrics.json 檔案...")
    val_metrics = list(base_path.glob("**/val_metrics.json"))
    print(f"找到 {len(val_metrics)} 個檔案")

    for file_path in val_metrics:
        result = process_experiment_file(file_path, base_path, "val_metrics")
        if result:
            all_experiments.append(result)

    # 4. 處理 Agent metrics.json 檔案
    print("\n[4/4] 處理 Agent metrics.json 檔案...")
    agent_metrics_files = [
        base_path / "2080_LLM/Refactored_Psy/psy-ref-repos/Criteria_Evidence_Agent/metrics.json",
        base_path / "3090_LLM/Criteria_Evidence_Agent/metrics.json"
    ]

    for file_path in agent_metrics_files:
        if file_path.exists():
            result = process_experiment_file(file_path, base_path, "agent_metrics")
            if result:
                all_experiments.append(result)

    print(f"\n總共成功處理了 {len(all_experiments)} 個實驗")

    # 儲存詳細實驗列表
    output_file = base_path / "all_experiments.json"
    print(f"\n儲存詳細實驗列表到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_experiments, f, indent=2, ensure_ascii=False)

    # 找出最佳實驗
    print("\n計算最佳實驗...")
    best_experiments = find_best_experiments(all_experiments)
    print(f"找到 {len(best_experiments)} 組最佳實驗")

    # 按 ranking_criterion 排序
    best_experiments.sort(key=lambda x: (
        x['task_type'],
        x['model_family'],
        x['augmentation'],
        -x['best_metrics'].get(x['ranking_criterion'], 0)
    ))

    # 儲存最佳實驗彙總
    summary_file = base_path / "best_experiments_summary.json"
    print(f"儲存最佳實驗彙總到: {summary_file}")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(best_experiments, f, indent=2, ensure_ascii=False)

    # 輸出統計資訊
    print("\n" + "="*60)
    print("統計資訊")
    print("="*60)
    print(f"總實驗數: {len(all_experiments)}")
    print(f"最佳實驗組數: {len(best_experiments)}")

    # 按任務類型統計
    task_counts = defaultdict(int)
    for exp in all_experiments:
        task_counts[exp["task_type"]] += 1

    print("\n按任務類型統計:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task:40s}: {count:4d}")

    # 按模型家族統計
    model_counts = defaultdict(int)
    for exp in all_experiments:
        model_counts[exp["model_family"]] += 1

    print("\n按模型家族統計:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"  {model:40s}: {count:4d}")

    # 按增強類型統計
    aug_counts = defaultdict(int)
    for exp in all_experiments:
        aug_counts[exp["augmentation"]] += 1

    print("\n按增強類型統計:")
    for aug, count in sorted(aug_counts.items(), key=lambda x: -x[1]):
        print(f"  {aug:40s}: {count:4d}")

    # 按 GPU 統計
    gpu_counts = defaultdict(int)
    for exp in all_experiments:
        gpu = exp["project_path"].split("/")[0]
        gpu_counts[gpu] += 1

    print("\n按 GPU 機器統計:")
    for gpu, count in sorted(gpu_counts.items()):
        print(f"  {gpu:40s}: {count:4d}")

    # 顯示一些最佳實驗樣本
    print("\n" + "="*60)
    print("最佳實驗樣本 (前10組)")
    print("="*60)
    for i, exp in enumerate(best_experiments[:10]):
        print(f"\n[{i+1}] {exp['group_key']}")
        print(f"    專案: {exp['project_path']}")
        print(f"    實驗ID: {exp.get('best_experiment_id', 'N/A')}")
        metric_value = exp['best_metrics'].get(exp['ranking_criterion'], 0)
        if metric_value != 0:
            print(f"    排名指標: {exp['ranking_criterion']} = {metric_value:.4f}")
        else:
            print(f"    排名指標: {exp['ranking_criterion']} = N/A")
        print(f"    該組實驗數: {exp['num_experiments_in_group']}")

    print("\n完成！")


if __name__ == "__main__":
    main()
