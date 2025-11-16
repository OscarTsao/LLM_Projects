# LLM_Projects 實驗結果分析彙總

## 分析概況

**分析日期**: 2025-11-15  
**分析對象**: LLM_Projects 目錄中所有 evaluation_report.json  
**檔案總數**: 163 個實驗  
**項目數**: 3 個

---

## 項目清單及統計

### 1. 2080_LLM/DataAug_DeBERTa_Evidence
- **實驗數量**: 81
- **任務類型**: Evidence Sentence Classification
- **模型族系**: DeBERTa
- **單/多任務**: Single Task
- **數據增強方法**: 
  - Case Variation (75 個實驗 - 92.6%)
  - No Change (6 個實驗 - 7.4%)
- **最佳實驗**: trial_0021
- **最佳性能指標**:
  - Evidence Macro F1: 0.4571
  - Evidence Accuracy: 0.5714
  - Overall Macro F1 Mean: 0.2286
  - Validation Evidence Macro F1: 0.3556

### 2. 2080_LLM/DataAug_Multi_Evidence
- **實驗數量**: 1
- **任務類型**: Multi-Task with Evidence
- **模型族系**: Unknown
- **單/多任務**: Multi Task
- **數據增強方法**: None
- **最佳實驗**: trial_123e4567-e89b-12d3-a456-426614174000
- **最佳性能指標**:
  - Span F1: 0.0
  - Precision: 0.0
  - Recall: 0.0
  - (注: 該實驗結果全為 0，可能尚未完成或失敗)

### 3. 4090_LLM/DataAug_DeBERTa_Evidence
- **實驗數量**: 81
- **任務類型**: Evidence Sentence Classification
- **模型族系**: DeBERTa
- **單/多任務**: Single Task
- **數據增強方法**:
  - Case Variation (75 個實驗 - 92.6%)
  - No Change (6 個實驗 - 7.4%)
- **最佳實驗**: trial_0021
- **最佳性能指標**:
  - Evidence Macro F1: 0.4571
  - Evidence Accuracy: 0.5714
  - Overall Macro F1 Mean: 0.2286
  - Validation Evidence Macro F1: 0.3556

---

## 主要發現

### 1. 數據增強效果
- **Case Variation** 是主要使用的增強方法，共 150 個實驗 (92.0%)
- 在 Evidence 任務上，Case Variation 與 No Change 相比，性能相近或略優
- 最佳實驗普遍採用 Case Variation

### 2. 模型表現
- **Evidence Classification 任務**
  - 最佳 Macro F1: 0.4571 (Trial 0021 和 0119)
  - 最佳 Accuracy: 0.5714
  - 驗證集表現: 0.3556 (可能存在過擬合)

- **Criteria Classification 任務**
  - 性能較低，大多數實驗的 Macro F1 < 0.2
  - 有些實驗的 F1 為 0

### 3. GPU 對比
- 2080_LLM 和 4090_LLM 的結果完全相同，表明實驗是在兩個 GPU 上重複運行

### 4. Multi-Task 實驗
- Multi-Task 實驗目前只有 1 個，且性能為 0
- 該項目可能需要進一步調試

---

## 前 10 個最佳實驗 (Evidence F1 排序)

1. trial_0021 (2080/4090) - Case Variation - F1: 0.4571
2. trial_0119 (2080/4090) - Case Variation - F1: 0.4571
3. trial_0013 (2080/4090) - Case Variation - F1: 0.3333
4. trial_0006 (2080/4090) - Case Variation - F1: 0.3333
5. trial_0115 (2080/4090) - Case Variation - F1: 0.3175
6. trial_0011 (2080/4090) - No Change - F1: 0.2619
7. trial_0058 (2080/4090) - Case Variation - F1: 0.2000
8. trial_0007 (2080/4090) - Case Variation - F1: 0.2000
9. trial_0023 (2080/4090) - Case Variation - F1: 0.2000
10. trial_0002 (2080/4090) - Case Variation - F1: 0.2000

---

## 數據文件說明

### 已生成的分析檔案

1. **EXPERIMENTS_ANALYSIS_COMPLETE.json** (192 KB)
   - 完整的 JSON 格式分析結果
   - 包含每個項目的所有實驗詳細信息
   - 包括最佳實驗和前 10 名排序

2. **EXPERIMENTS_SUMMARY.csv** (33 KB)
   - CSV 格式的實驗摘要表
   - 所有 163 個實驗的關鍵指標
   - 便於在 Excel 或其他工具中進一步分析

3. **EXPERIMENTS_ANALYSIS_REPORT.txt** (4.1 KB)
   - 人類可讀的詳細文本報告
   - 包含統計信息和最佳實驗詳情

4. **file_manifest.json** (74 KB)
   - 所有 163 個 evaluation_report.json 檔案的完整清單
   - 包含完整路徑和相對路徑信息

5. **ANALYSIS_SUMMARY.md** (本文件)
   - 分析的高層概述

---

## 建議和後續步驟

### 1. Evidence Classification 改進
- 目前最佳 F1 達到 0.4571，可考慮：
  - 調整超參數 (學習率、批量大小、epoch 數)
  - 嘗試其他數據增強策略 (不僅是 case variation)
  - 使用更大的模型或預訓練模型

### 2. Criteria Classification 改進
- 目前表現較差，需要重點關注
- 建議檢查訓練數據的質量和平衡性

### 3. Multi-Task 調試
- 單一的 multi-task 實驗失敗，需要調查根本原因
- 建議進行更多的 multi-task 實驗

### 4. 數據增強策略
- Case variation 為主要方法，但仍有改進空間
- 建議嘗試：
  - 語義相似的詞替換 (如 synonym replacement)
  - 反向翻譯 (back-translation)
  - LLM 生成的改寫 (LLM paraphrase)

---

## 詳細數據訪問

### 使用 Python 讀取完整數據

```python
import json

# 讀取完整分析結果
with open('EXPERIMENTS_ANALYSIS_COMPLETE.json') as f:
    data = json.load(f)

# 訪問特定項目的最佳實驗
project = '2080_LLM/DataAug_DeBERTa_Evidence'
best_exp = data['projects'][project]['best_experiment_details']
print(f"最佳實驗: {best_exp['experiment_id']}")
print(f"性能: {best_exp['metrics']}")
```

### 使用 CSV 進行比較分析

CSV 檔案包含所有實驗的關鍵指標，可直接在 Excel、Pandas 等工具中進行：
- 數據透視表分析
- 趨勢圖表繪製
- 統計對比

---

**分析完成**

所有分析檔案已保存至 `/home/user/LLM_Projects/` 目錄。
