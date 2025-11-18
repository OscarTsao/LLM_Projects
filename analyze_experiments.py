#!/usr/bin/env python3
"""
實驗結果分析腳本
分析 LLM_Projects 目錄中的所有實驗結果
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict


def extract_project_info(project_path: str) -> Dict[str, Any]:
    """從專案路徑推斷專案資訊"""
    project_name = Path(project_path).name
    parent_dir = Path(project_path).parent.name

    # 推斷 GPU 類型
    gpu_match = re.search(r'(2080|3090|4070ti|4090)', parent_dir)
    gpu_type = gpu_match.group(1) if gpu_match else "unknown"

    # 推斷模型家族
    model_family = "unknown"
    if "DeBERTa" in project_name or "deberta" in project_name.lower():
        model_family = "deberta_v3"
    elif "Mental" in project_name or "mental" in project_name.lower():
        model_family = "mental_bert"
    elif "BERT" in project_name or "bert" in project_name.lower():
        model_family = "bert"
    elif "RoBERTa" in project_name or "roberta" in project_name.lower():
        model_family = "roberta"

    # 推斷任務類型
    task_type = "unknown"
    single_vs_multi = "unknown"

    if "Criteria" in project_name and "Evidence" in project_name:
        task_type = "multi_task_criteria_evidence"
        single_vs_multi = "multi"
    elif "Criteria" in project_name:
        task_type = "criteria_matching"
        single_vs_multi = "single"
    elif "Evidence" in project_name:
        # 需要看 metrics 來區分是 sentence 還是 span
        task_type = "evidence_detection"  # 暫時標記，稍後根據 metrics 調整
        single_vs_multi = "single"
    elif "Risk" in project_name:
        task_type = "risk_detection"
        single_vs_multi = "single"
    elif "Rerank" in project_name:
        task_type = "reranker"
        single_vs_multi = "single"

    # 推斷資料增強類型
    augmentation = "unknown"
    if "NoAug" in project_name:
        augmentation = "none"
    elif "DataAug" in project_name:
        # 需要從 config 或其他地方推斷具體類型
        augmentation = "hybrid"  # 預設為 hybrid，後續可能調整
    elif "EDA" in project_name:
        augmentation = "eda"
    elif "BackTrans" in project_name:
        augmentation = "backtranslation"

    # 推斷資料版本
    data_version = "unknown"
    if "redsm5" in project_name.lower() or "REDSM5" in project_name:
        data_version = "redsm5_v2"

    return {
        "gpu_type": gpu_type,
        "model_family": model_family,
        "task_type": task_type,
        "single_vs_multi": single_vs_multi,
        "augmentation": augmentation,
        "data_version": data_version
    }


def infer_augmentation_from_trial(trial_path: Path) -> str:
    """從 trial 目錄中推斷增強類型"""
    aug_dir = trial_path / "augmentation"
    if aug_dir.exists():
        aug_file = aug_dir / "augmentation_samples.jsonl"
        if aug_file.exists():
            try:
                with open(aug_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if first_line:
                        sample = json.loads(first_line)
                        # 可以根據 sample 的特徵推斷增強類型
                        # 目前先返回 "textattack" 作為預設
                        return "textattack"
            except:
                pass
    return "unknown"


def extract_metrics_from_evaluation_report(report: Dict) -> Dict[str, Any]:
    """從 evaluation_report.json 提取指標"""
    metrics = {}

    # 檢查是否有 test_metrics
    if "test_metrics" in report:
        test_metrics = report["test_metrics"]

        # 多任務格式 (criteria + evidence)
        if "test_cri_macro_f1" in test_metrics:
            metrics["criteria_macro_f1"] = test_metrics.get("test_cri_macro_f1")
            metrics["criteria_accuracy"] = test_metrics.get("test_cri_accuracy")
            metrics["evidence_macro_f1"] = test_metrics.get("test_ev_macro_f1")
            metrics["evidence_accuracy"] = test_metrics.get("test_ev_accuracy")
            metrics["macro_f1_mean"] = test_metrics.get("test_macro_f1_mean")

        # Span 抽取格式
        elif "evidence_binding" in test_metrics:
            eb = test_metrics["evidence_binding"]
            metrics["span_f1"] = eb.get("span_f1")
            metrics["precision"] = eb.get("precision")
            metrics["recall"] = eb.get("recall")
            metrics["exact_match"] = eb.get("exact_match")
            metrics["char_f1"] = eb.get("char_f1")

        # 單一分類任務格式
        elif "macro_f1" in test_metrics:
            metrics["macro_f1"] = test_metrics.get("macro_f1")
            metrics["micro_f1"] = test_metrics.get("micro_f1")
            metrics["accuracy"] = test_metrics.get("accuracy")
            metrics["roc_auc"] = test_metrics.get("roc_auc")
            metrics["pr_auc"] = test_metrics.get("pr_auc")

    # 檢查是否有 training_metrics (用於推斷訓練資訊)
    training_info = {}
    if "training_metrics" in report:
        # 可以提取驗證集最佳指標
        pass

    return {
        "metrics": metrics,
        "training_info": training_info
    }


def process_evaluation_report(file_path: Path, base_path: Path) -> Optional[Dict]:
    """處理單個 evaluation_report.json 檔案"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        # 獲取相對路徑
        rel_path = file_path.relative_to(base_path)
        if len(rel_path.parts) > 1:
            project_path = str(Path(rel_path.parts[0]) / rel_path.parts[1])
        else:
            project_path = str(rel_path.parts[0])

        # 提取專案資訊
        project_info = extract_project_info(str(file_path.parent.parent.parent))

        # 提取實驗 ID
        experiment_id = file_path.parent.name

        # 推斷增強類型（如果還是 unknown）
        if project_info["augmentation"] in ["unknown", "hybrid"]:
            aug_type = infer_augmentation_from_trial(file_path.parent)
            if aug_type != "unknown":
                project_info["augmentation"] = aug_type

        # 提取指標
        metric_data = extract_metrics_from_evaluation_report(report)

        # 根據指標調整任務類型
        if project_info["task_type"] == "evidence_detection":
            if "span_f1" in metric_data["metrics"]:
                project_info["task_type"] = "evidence_span"
            elif "evidence_macro_f1" in metric_data["metrics"]:
                project_info["task_type"] = "evidence_sentence"

        # 組合結果
        result = {
            "project_path": project_path,
            "experiment_id": experiment_id,
            "task_type": project_info["task_type"],
            "model_family": project_info["model_family"],
            "single_vs_multi_task": project_info["single_vs_multi"],
            "data_version": project_info["data_version"],
            "augmentation": project_info["augmentation"],
            "metrics": metric_data["metrics"],
            "training_info": metric_data["training_info"],
            "file_path": str(file_path)
        }

        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_mlflow_metrics(file_path: Path, base_path: Path) -> Optional[Dict]:
    """處理 MLflow test_metrics.json 檔案"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        # 獲取相對路徑和專案資訊
        rel_path = file_path.relative_to(base_path)
        if len(rel_path.parts) > 1:
            project_path = str(Path(rel_path.parts[0]) / rel_path.parts[1])
        else:
            project_path = str(rel_path.parts[0])

        # 提取 run_id
        run_id = file_path.parent.parent.name

        # 提取專案資訊
        project_info = extract_project_info(str(file_path.parent.parent.parent.parent.parent))

        result = {
            "project_path": project_path,
            "experiment_id": run_id,
            "task_type": project_info["task_type"],
            "model_family": project_info["model_family"],
            "single_vs_multi_task": project_info["single_vs_multi"],
            "data_version": project_info["data_version"],
            "augmentation": project_info["augmentation"],
            "metrics": metrics,
            "training_info": {},
            "file_path": str(file_path)
        }

        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_agent_metrics(file_path: Path, base_path: Path) -> Optional[Dict]:
    """處理 Criteria_Evidence_Agent metrics.json 檔案"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        rel_path = file_path.relative_to(base_path)
        if len(rel_path.parts) > 1:
            project_path = str(Path(rel_path.parts[0]) / rel_path.parts[1])
        else:
            project_path = str(rel_path.parts[0])

        project_info = extract_project_info(str(file_path.parent))

        result = {
            "project_path": project_path,
            "experiment_id": "agent_run",
            "task_type": "agent_multi_task",
            "model_family": project_info["model_family"],
            "single_vs_multi_task": "multi",
            "data_version": project_info["data_version"],
            "augmentation": "none",
            "metrics": metrics,
            "training_info": {},
            "file_path": str(file_path)
        }

        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def find_best_experiments(all_experiments: List[Dict]) -> List[Dict]:
    """找出每個組合的最佳實驗"""
    # 按照 task_type × model_family × single_vs_multi × augmentation 分組
    groups = defaultdict(list)

    for exp in all_experiments:
        # 跳過沒有有效指標的實驗
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

        if task_type in ["evidence_span"]:
            ranking_key = "span_f1"
        elif task_type == "reranker":
            ranking_key = "ndcg"
        elif task_type == "multi_task_criteria_evidence":
            ranking_key = "macro_f1_mean"
        else:
            # 優先使用 micro_f1, 然後 macro_f1
            ranking_key = "micro_f1"
            # 檢查是否有 micro_f1
            if not any(ranking_key in exp["metrics"] for exp in experiments):
                ranking_key = "macro_f1"
            if not any(ranking_key in exp["metrics"] for exp in experiments):
                ranking_key = "accuracy"

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
                "ranking_criterion": ranking_key
            })

    return best_experiments


def main():
    """主函數"""
    base_path = Path("/home/user/LLM_Projects")
    all_experiments = []

    print("開始處理實驗檔案...")

    # 處理 evaluation_report.json 檔案
    print("\n處理 evaluation_report.json 檔案...")
    eval_reports = list(base_path.glob("**/evaluation_report.json"))
    print(f"找到 {len(eval_reports)} 個 evaluation_report.json 檔案")

    for i, file_path in enumerate(eval_reports):
        if i % 20 == 0:
            print(f"處理進度: {i}/{len(eval_reports)}")
        result = process_evaluation_report(file_path, base_path)
        if result:
            all_experiments.append(result)

    # 處理 MLflow test_metrics.json 檔案
    print("\n處理 MLflow test_metrics.json 檔案...")
    mlflow_metrics = list(base_path.glob("**/mlruns/**/artifacts/test_metrics.json"))
    print(f"找到 {len(mlflow_metrics)} 個 test_metrics.json 檔案")

    for file_path in mlflow_metrics:
        result = process_mlflow_metrics(file_path, base_path)
        if result:
            all_experiments.append(result)

    # 處理 Agent metrics.json 檔案
    print("\n處理 Criteria_Evidence_Agent metrics.json 檔案...")
    agent_metrics = [
        base_path / "2080_LLM/Refactored_Psy/psy-ref-repos/Criteria_Evidence_Agent/metrics.json",
        base_path / "3090_LLM/Criteria_Evidence_Agent/metrics.json"
    ]

    for file_path in agent_metrics:
        if file_path.exists():
            result = process_agent_metrics(file_path, base_path)
            if result:
                all_experiments.append(result)

    print(f"\n總共處理了 {len(all_experiments)} 個實驗")

    # 儲存詳細實驗列表
    output_file = base_path / "all_experiments.json"
    print(f"\n儲存詳細實驗列表到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_experiments, f, indent=2, ensure_ascii=False)

    # 找出最佳實驗
    print("\n計算最佳實驗...")
    best_experiments = find_best_experiments(all_experiments)
    print(f"找到 {len(best_experiments)} 組最佳實驗")

    # 儲存最佳實驗彙總
    summary_file = base_path / "best_experiments_summary.json"
    print(f"儲存最佳實驗彙總到: {summary_file}")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(best_experiments, f, indent=2, ensure_ascii=False)

    # 輸出統計資訊
    print("\n=== 統計資訊 ===")
    print(f"總實驗數: {len(all_experiments)}")
    print(f"最佳實驗組數: {len(best_experiments)}")

    # 按任務類型統計
    task_counts = defaultdict(int)
    for exp in all_experiments:
        task_counts[exp["task_type"]] += 1

    print("\n按任務類型統計:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    # 按模型家族統計
    model_counts = defaultdict(int)
    for exp in all_experiments:
        model_counts[exp["model_family"]] += 1

    print("\n按模型家族統計:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}")

    # 按增強類型統計
    aug_counts = defaultdict(int)
    for exp in all_experiments:
        aug_counts[exp["augmentation"]] += 1

    print("\n按增強類型統計:")
    for aug, count in sorted(aug_counts.items()):
        print(f"  {aug}: {count}")

    print("\n完成！")


if __name__ == "__main__":
    main()
