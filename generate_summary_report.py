#!/usr/bin/env python3
"""
生成實驗分析摘要報告
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    base_path = Path("/home/user/LLM_Projects")

    # 讀取實驗數據
    with open(base_path / "all_experiments.json") as f:
        all_experiments = json.load(f)

    with open(base_path / "best_experiments_summary.json") as f:
        best_experiments = json.load(f)

    print("="*80)
    print("LLM 實驗結果分析摘要報告")
    print("="*80)
    print(f"\n總實驗數: {len(all_experiments)}")
    print(f"最佳實驗組數: {len(best_experiments)}")

    # === 1. 多任務實驗分析 ===
    print("\n" + "="*80)
    print("1. 多任務實驗 (Criteria + Evidence) - Top 10")
    print("="*80)

    multi_task_exps = [e for e in all_experiments
                       if e['task_type'] == 'multi_task_criteria_evidence'
                       and e['metrics'].get('macro_f1_mean')]

    multi_task_exps.sort(key=lambda x: x['metrics']['macro_f1_mean'], reverse=True)

    print(f"\n找到 {len(multi_task_exps)} 個多任務實驗")
    print(f"模型: {multi_task_exps[0]['model_family'] if multi_task_exps else 'N/A'}")
    print(f"數據增強: {multi_task_exps[0]['augmentation'] if multi_task_exps else 'N/A'}")
    print("\nTop 10 實驗:")
    print(f"{'排名':<6} {'實驗ID':<15} {'Mean F1':<10} {'Criteria F1':<12} {'Evidence F1':<12} {'GPU':<10}")
    print("-"*80)

    for i, exp in enumerate(multi_task_exps[:10]):
        gpu = exp['project_path'].split('/')[0]
        print(f"{i+1:<6} {exp['experiment_id']:<15} "
              f"{exp['metrics']['macro_f1_mean']:<10.4f} "
              f"{exp['metrics']['criteria_macro_f1']:<12.4f} "
              f"{exp['metrics']['evidence_macro_f1']:<12.4f} "
              f"{gpu:<10}")

    # === 2. 單任務分類實驗 ===
    print("\n" + "="*80)
    print("2. 單任務分類實驗 (DataAugmentation_Evaluation) - Top 10")
    print("="*80)

    eval_exps = [e for e in all_experiments
                 if 'DataAugmentation_Evaluation' in e['project_path']
                 and e['metrics'].get('f1')]

    eval_exps.sort(key=lambda x: x['metrics']['f1'], reverse=True)

    print(f"\n找到 {len(eval_exps)} 個實驗")
    print("\nTop 10 實驗:")
    print(f"{'排名':<6} {'實驗ID':<25} {'F1':<10} {'Accuracy':<10} {'ROC-AUC':<10}")
    print("-"*80)

    for i, exp in enumerate(eval_exps[:10]):
        print(f"{i+1:<6} {exp['experiment_id']:<25} "
              f"{exp['metrics']['f1']:<10.4f} "
              f"{exp['metrics'].get('accuracy', 0):<10.4f} "
              f"{exp['metrics'].get('roc_auc', 0):<10.4f}")

    # === 3. Baseline 實驗 (No Augmentation) ===
    print("\n" + "="*80)
    print("3. Baseline 實驗 (無數據增強) - Criteria_Baseline_5Fold_NoAug")
    print("="*80)

    baseline_exps = [e for e in all_experiments
                     if 'Criteria_Baseline_5Fold_NoAug' in e['project_path']]

    print(f"\n找到 {len(baseline_exps)} 個實驗")

    # 計算平均值
    if baseline_exps:
        avg_f1_macro = sum(e['metrics'].get('f1_macro', 0) for e in baseline_exps) / len(baseline_exps)
        avg_accuracy = sum(e['metrics'].get('accuracy', 0) for e in baseline_exps) / len(baseline_exps)
        avg_auc = sum(e['metrics'].get('auc', 0) for e in baseline_exps) / len(baseline_exps)

        print(f"\n平均指標:")
        print(f"  F1 Macro:  {avg_f1_macro:.4f}")
        print(f"  Accuracy:  {avg_accuracy:.4f}")
        print(f"  AUC:       {avg_auc:.4f}")

        print(f"\n各 Fold 詳細結果:")
        print(f"{'實驗ID':<50} {'F1 Macro':<12} {'Accuracy':<12} {'AUC':<10}")
        print("-"*80)
        for exp in baseline_exps[:10]:
            print(f"{exp['experiment_id']:<50} "
                  f"{exp['metrics'].get('f1_macro', 0):<12.4f} "
                  f"{exp['metrics'].get('accuracy', 0):<12.4f} "
                  f"{exp['metrics'].get('auc', 0):<10.4f}")

    # === 4. 按 GPU 分布統計 ===
    print("\n" + "="*80)
    print("4. GPU 機器實驗分布統計")
    print("="*80)

    gpu_stats = defaultdict(lambda: {'count': 0, 'tasks': defaultdict(int)})

    for exp in all_experiments:
        gpu = exp['project_path'].split('/')[0]
        gpu_stats[gpu]['count'] += 1
        gpu_stats[gpu]['tasks'][exp['task_type']] += 1

    print(f"\n{'GPU':<15} {'總實驗數':<12} {'主要任務類型':<40}")
    print("-"*80)

    for gpu in sorted(gpu_stats.keys()):
        stats = gpu_stats[gpu]
        main_task = max(stats['tasks'].items(), key=lambda x: x[1]) if stats['tasks'] else ('N/A', 0)
        print(f"{gpu:<15} {stats['count']:<12} {main_task[0]:<40} ({main_task[1]} exps)")

    # === 5. 數據增強策略效果對比 ===
    print("\n" + "="*80)
    print("5. 數據增強策略效果對比")
    print("="*80)

    aug_performance = defaultdict(list)

    for exp in multi_task_exps:
        aug = exp['augmentation']
        f1_mean = exp['metrics']['macro_f1_mean']
        aug_performance[aug].append(f1_mean)

    print(f"\n多任務實驗中各增強策略的平均 F1:")
    print(f"{'增強策略':<15} {'實驗數':<10} {'平均 F1':<12} {'最佳 F1':<12}")
    print("-"*80)

    for aug in sorted(aug_performance.keys()):
        scores = aug_performance[aug]
        avg_f1 = sum(scores) / len(scores)
        best_f1 = max(scores)
        print(f"{aug:<15} {len(scores):<10} {avg_f1:<12.4f} {best_f1:<12.4f}")

    # === 6. 關鍵發現與建議 ===
    print("\n" + "="*80)
    print("6. 關鍵發現與建議")
    print("="*80)

    print("\n【關鍵發現】")

    if multi_task_exps:
        best_multi = multi_task_exps[0]
        print(f"\n1. 最佳多任務模型:")
        print(f"   - 實驗ID: {best_multi['experiment_id']}")
        print(f"   - Mean F1: {best_multi['metrics']['macro_f1_mean']:.4f}")
        print(f"   - Criteria F1: {best_multi['metrics']['criteria_macro_f1']:.4f}")
        print(f"   - Evidence F1: {best_multi['metrics']['evidence_macro_f1']:.4f}")
        print(f"   - 模型: {best_multi['model_family']}")
        print(f"   - 增強: {best_multi['augmentation']}")

    if eval_exps:
        best_single = eval_exps[0]
        print(f"\n2. 最佳單任務分類模型:")
        print(f"   - 實驗ID: {best_single['experiment_id']}")
        print(f"   - F1: {best_single['metrics']['f1']:.4f}")
        print(f"   - Accuracy: {best_single['metrics'].get('accuracy', 0):.4f}")
        print(f"   - ROC-AUC: {best_single['metrics'].get('roc_auc', 0):.4f}")

    print(f"\n3. 數據增強效果:")
    if 'hybrid' in aug_performance and 'none' in aug_performance:
        hybrid_avg = sum(aug_performance['hybrid']) / len(aug_performance['hybrid'])
        none_avg = sum(aug_performance['none']) / len(aug_performance['none']) if aug_performance['none'] else 0
        if none_avg > 0:
            improvement = ((hybrid_avg - none_avg) / none_avg) * 100
            print(f"   - Hybrid 增強相比無增強提升: {improvement:.2f}%")

    print(f"\n4. 模型表現分布:")
    if multi_task_exps:
        all_f1s = [e['metrics']['macro_f1_mean'] for e in multi_task_exps]
        print(f"   - 最高 F1: {max(all_f1s):.4f}")
        print(f"   - 平均 F1: {sum(all_f1s)/len(all_f1s):.4f}")
        print(f"   - 最低 F1: {min(all_f1s):.4f}")
        print(f"   - 標準差: {(sum((x - sum(all_f1s)/len(all_f1s))**2 for x in all_f1s) / len(all_f1s))**0.5:.4f}")

    print("\n【建議】")
    print("\n1. 進一步超參數優化可能提升最佳模型表現")
    print("2. 考慮 ensemble 多個高分模型以提高穩定性")
    print("3. 分析失敗案例以理解模型弱點")
    print("4. 評估計算成本與性能的權衡")

    print("\n" + "="*80)
    print("報告生成完成")
    print("="*80)

    # 保存報告到文件
    report_file = Path("/home/user/LLM_Projects/EXPERIMENT_ANALYSIS_REPORT.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        # (可以在這裡重定向 stdout 來保存報告)
        pass

    print(f"\n詳細數據已保存到:")
    print(f"  - all_experiments.json")
    print(f"  - best_experiments_summary.json")


if __name__ == "__main__":
    main()
