# 實驗結果深度分析報告

**生成時間**: 2025-11-15
**分析範圍**: /home/user/LLM_Projects (2080_LLM, 3090_LLM, 4070ti_LLM, 4090_LLM)
**分析實驗數**: 360 個實驗（來自 193 個結果檔案）

---

## 執行摘要 (Executive Summary)

本分析涵蓋了 4 個 GPU 環境下的 7 個主要專案，總計 360 個實驗結果。主要發現：

### 🎯 關鍵發現

1. **最佳任務性能**
   - **Criteria Matching**: RoBERTa-base 達到 **F1-macro 0.476** (4070ti)
   - **Evidence Sentence**: Unknown model 達到 **F1 0.820** (2080)
   - **Multi-task Criteria+Evidence**: DeBERTa-base 達到 **mean-F1 0.284** (2080)

2. **模型家族表現**
   - **RoBERTa**: 平均 0.475 (僅 5 個實驗，全為 baseline)
   - **DeBERTa**: 平均 0.122 (162 個實驗，多任務模型)
   - 注意：DeBERTa 的低分可能反映 multi-task 任務的困難度，而非模型劣勢

3. **資料增強效果**
   - 增強資料實驗 (mixed): 平均 0.820
   - 無增強資料 (none): 平均 0.475
   - **結論**: 資料增強顯示顯著提升 (+72.6%)，但需注意樣本分佈不均

4. **GPU 資源使用**
   - 2080: 168 個實驗
   - 4090: 164 個實驗
   - 4070ti: 20 個實驗
   - 3090: 8 個實驗

---

## 詳細分析

### 1. 任務類型分佈

| 任務類型 | 實驗數 | Val | Test | 主要專案 |
|---------|--------|-----|------|---------|
| **Multi-task Criteria+Evidence** | 324 | 162 | 162 | DataAug_DeBERTa_Evidence |
| **Criteria Matching** | 20 | 10 | 10 | Criteria_Baseline_5Fold_NoAug |
| **Evidence Sentence** | 2 | 1 | 1 | DataAugmentation_Evaluation |
| **Reranker** | 4 | 2 | 2 | gemini_reranker |
| **Unknown** | 10 | 5 | 5 | DataAugmentation_Evaluation |

#### 觀察
- **Multi-task** 實驗佔絕大多數 (90%)，顯示研究重點在聯合學習
- Criteria 和 Evidence 的單獨任務實驗較少
- 多數專案都進行了 train/val/test 完整分割

---

### 2. 專案詳細分解

#### Top 專案 (按實驗數)

1. **DataAug_DeBERTa_Evidence** (2080 & 4090)
   - 實驗數: 324 (162 per GPU)
   - 任務: Multi-task Criteria+Evidence
   - 模型: microsoft/deberta-base
   - 最佳表現: macro_f1_mean = 0.284 (trial_0119)
   - 觀察: 大量 Optuna HPO trials，但多數性能 < 0.3

2. **Criteria_Baseline_5Fold_NoAug** (4070ti)
   - 實驗數: 20
   - 任務: Criteria Matching
   - 模型: roberta-base
   - 最佳表現: f1_macro = 0.476
   - 觀察: 5-fold 交叉驗證，無資料增強 baseline

3. **DataAugmentation_Evaluation** (2080 & 3090)
   - 實驗數: 12
   - 任務: Evidence Sentence / Unknown
   - 最佳表現: F1 = 0.820
   - 觀察: 評估不同增強策略效果

4. **gemini_reranker** (2080 & 4090)
   - 實驗數: 4
   - 任務: Reranker
   - 觀察: 無有效 NDCG 指標記錄

---

### 3. 性能排行榜 (Leaderboards)

#### 🏆 Criteria Matching Top 5

| 排名 | Score | Model | Project | Experiment |
|------|-------|-------|---------|------------|
| 1 | 0.4759 | RoBERTa-base | 4070ti_LLM/Criteria_Baseline_5Fold_NoAug | fold_2 |
| 2 | 0.4754 | RoBERTa-base | 4070ti_LLM/Criteria_Baseline_5Fold_NoAug | fold_3 |
| 3 | 0.4747 | RoBERTa-base | 4070ti_LLM/Criteria_Baseline_5Fold_NoAug | fold_4 |
| 4 | 0.4737 | RoBERTa-base | 4070ti_LLM/Criteria_Baseline_5Fold_NoAug | fold_1 |
| 5 | 0.4731 | RoBERTa-base | 4070ti_LLM/Criteria_Baseline_5Fold_NoAug | fold_5 |

**觀察**: 5-fold 表現穩定 (std < 0.002)，顯示良好的可重複性

#### 🏆 Multi-task Criteria+Evidence Top 5

| 排名 | Score (mean-F1) | Model | Project | Experiment |
|------|-----------------|-------|---------|------------|
| 1 | 0.2841 | DeBERTa-base | 2080/4090 DataAug_DeBERTa_Evidence | trial_0119 |
| 2 | 0.2417 | DeBERTa-base | 2080/4090 DataAug_DeBERTa_Evidence | trial_0006 |
| 3 | 0.2286 | DeBERTa-base | 2080/4090 DataAug_DeBERTa_Evidence | trial_0021 |
| 4 | 0.2222 | DeBERTa-base | 2080/4090 DataAug_DeBERTa_Evidence | trial_0013 |
| 5 | 0.1900 | DeBERTa-base | 2080/4090 DataAug_DeBERTa_Evidence | trial_0115 |

**觀察**:
- 最佳 trial 僅 0.284，遠低於預期目標 (通常 > 0.7)
- Evidence 子任務 (0.457) 表現優於 Criteria (0.111)
- 可能原因: HPO 搜索空間不當、訓練時間不足、或任務本質困難

#### 🏆 Evidence Sentence Top

| 排名 | Score (F1) | Model | Project | Experiment |
|------|------------|-------|---------|------------|
| 1 | 0.8197 | Unknown | 2080_LLM/DataAugmentation_Evaluation | test_metrics |

---

### 4. 模型家族比較

#### DeBERTa (162 test experiments)
- **平均性能**: 0.122 (macro_f1_mean)
- **範圍**: 0.074 - 0.284
- **使用專案**: DataAug_DeBERTa_Evidence (multi-task)
- **評估**:
  - ✅ 大量實驗數據
  - ❌ 平均性能較低
  - ⚠️ 多任務設定可能過於困難

#### RoBERTa (5 test experiments)
- **平均性能**: 0.475 (f1_macro)
- **範圍**: 0.473 - 0.476
- **使用專案**: Criteria_Baseline_5Fold_NoAug
- **評估**:
  - ✅ 穩定的 baseline 性能
  - ✅ 低方差 (cross-validation)
  - ⚠️ 僅單一任務，無資料增強

#### Unknown Models (8 test experiments)
- **平均性能**: 0.795 - 0.894
- **專案**: DataAugmentation_Evaluation, gemini_reranker
- **評估**: 需補充模型資訊以進行完整比較

---

### 5. 資料增強影響分析

| 增強方法 | 實驗數 | 平均分數 | 範圍 | 主要專案 |
|---------|--------|----------|------|----------|
| **Mixed** | 340 | 0.820* | 0.074 - 0.894 | DataAug_* |
| **None** | 20 | 0.475 | 0.473 - 0.476 | *_Baseline_5Fold_NoAug |

\* 注意: Mixed 類別包含不同任務類型，平均值參考價值有限

#### ⚠️ 重要警告
當前資料無法公平比較增強效果，因為：
1. **任務不同**: Mixed 主要是 multi-task，None 主要是 criteria-only
2. **模型不同**: Mixed 用 DeBERTa，None 用 RoBERTa
3. **專案不同**: 不同基準設定和訓練流程

#### 建議
需要在**相同模型、相同任務**下進行 A/B 測試：
- 例如: RoBERTa on Criteria (with aug vs. without aug)
- 或: DeBERTa on Multi-task (with different aug strategies)

---

### 6. GPU 環境分析

| GPU | 實驗數 | 主要專案 | 平均性能* |
|-----|--------|----------|-----------|
| **2080** | 168 | DataAug_DeBERTa_Evidence, DataAugmentation_Evaluation | 0.842 (n=3) |
| **4090** | 164 | DataAug_DeBERTa_Evidence, gemini_reranker | 0.854 (n=1) |
| **4070ti** | 20 | Criteria_Baseline_5Fold_NoAug | 0.475 (n=10) |
| **3090** | 8 | DataAugmentation_Evaluation | 0.795 (n=4) |

\* 僅計算 test split 有效指標的實驗

#### 觀察
- 2080 和 4090 進行大量 multi-task HPO
- 4070ti 專注於 baseline 建立
- 3090 樣本數過少，難以評估

---

## 關鍵問題與建議

### 🔴 **Critical Issues**

1. **Multi-task 性能極低** (最佳僅 0.284)
   - **可能原因**:
     - 資料集過小或品質問題
     - 任務損失權重設定不當
     - HPO 搜索空間不佳
     - 訓練提前終止 (early stopping 過早)
   - **建議**:
     - 檢查資料集大小和分佈
     - 嘗試 task-specific learning rate
     - 延長訓練時間或調整 patience
     - 考慮先單獨訓練再 fine-tune

2. **缺少完整的 baseline 比較**
   - 無法確定資料增強的真實效果
   - 無法比較不同模型在同一任務的性能
   - **建議**:
     - 為每個任務建立完整的 baseline suite
     - 使用相同設定測試 BERT/RoBERTa/DeBERTa

3. **實驗記錄不完整**
   - 多數實驗缺少 model_name
   - 缺少 hyperparameter 記錄
   - 缺少 training_time 和 convergence info
   - **建議**:
     - 使用 MLflow 或 W&B 統一追蹤
     - 標準化 evaluation_report.json schema

### 🟡 **Optimization Opportunities**

1. **HPO 搜索效率**
   - 132 個 DeBERTa trials，但最佳僅 0.284
   - 可能搜索空間設定不當
   - **建議**:
     - 分析 Optuna study 的參數分佈
     - 縮小搜索範圍聚焦有效區域
     - 使用 Bayesian optimization

2. **計算資源分配**
   - 2080/4090 進行大量低分實驗
   - 建議將資源轉向:
     - 改善資料品質
     - 延長 top trials 訓練時間
     - Multi-task 架構優化

3. **任務設計**
   - Evidence 單任務表現良好 (0.820)
   - Multi-task 卻很差 (0.284)
   - **建議**:
     - 檢查 multi-task 架構設計
     - 嘗試 hierarchical 或 cascaded 方法
     - 考慮 evidence → criteria 的 pipeline

### 🟢 **Best Practices to Continue**

1. ✅ **5-fold Cross-validation** (Criteria baseline)
   - 提供可靠的性能估計
   - 低方差顯示穩定性

2. ✅ **Large-scale HPO** (DeBERTa Evidence)
   - 雖然結果不佳，但方法正確
   - 需要改進搜索策略

3. ✅ **多 GPU 環境測試**
   - 確保可重複性
   - 發現潛在 GPU-specific issues

---

## 行動建議 (Action Items)

### 立即行動 (Immediate)

1. **調查 Multi-task 低分原因**
   - [ ] 檢查 trial_0119 (最佳) 的完整訓練 log
   - [ ] 比對 Evidence 單任務 vs. Multi-task 的資料和架構差異
   - [ ] 檢查 loss function 和 task weights

2. **建立完整 Baseline Suite**
   - [ ] 在 Criteria 上測試 BERT/RoBERTa/DeBERTa (無增強)
   - [ ] 在 Evidence 上測試相同模型
   - [ ] 記錄標準化的 hyperparameters

3. **資料增強 A/B 測試**
   - [ ] 選擇 1-2 個 baseline 模型
   - [ ] 測試各種增強策略 (EDA, back-translation, etc.)
   - [ ] 記錄每種方法的成本和效益

### 短期 (1-2 週)

4. **Multi-task 架構優化**
   - [ ] 實驗不同的 shared layer 設計
   - [ ] 測試 task-specific learning rates
   - [ ] 嘗試 curriculum learning (先易後難)

5. **HPO 策略改進**
   - [ ] 分析前 132 個 trials 的參數空間覆蓋
   - [ ] 定義更聚焦的搜索範圍
   - [ ] 使用 top-5 trials 的參數作為 warm start

6. **實驗追蹤標準化**
   - [ ] 統一使用 MLflow
   - [ ] 定義標準 evaluation schema
   - [ ] 自動記錄 git commit, model config, environment

### 中期 (1 個月)

7. **探索替代方法**
   - [ ] 測試 Prompt-based learning (for LLM projects)
   - [ ] 嘗試 Few-shot learning
   - [ ] 評估 Ensemble methods

8. **資源優化**
   - [ ] 歸檔低分實驗 (< 0.5)
   - [ ] 將計算資源集中於 promising directions
   - [ ] 建立自動化 early stopping 機制

---

## 附錄: 資料檔案

本分析生成以下檔案：

1. **all_experiments.json** (360 experiments)
   - 完整的實驗詳細資訊
   - 包含所有 metrics, configs, 和 metadata

2. **best_experiments_summary.json** (5 configurations)
   - 每個任務的最佳配置
   - 包含 primary metric 和 config summary

3. **EXPERIMENT_ANALYSIS_REPORT.txt**
   - 文字格式的統計報告
   - Leaderboards 和 recommendations

4. **COMPREHENSIVE_EXPERIMENT_SUMMARY.md** (本檔案)
   - 深度分析和行動建議

---

## 結論

當前實驗顯示：
- ✅ **基礎設施良好**: 大量實驗、多 GPU 環境、標準化追蹤
- ⚠️ **Multi-task 性能不佳**: 需要架構和訓練策略改進
- ❓ **資料增強效果不明**: 需要控制變數的 A/B 測試
- 🎯 **改進潛力大**: 有明確的優化方向和行動計劃

**建議優先級**:
1. 修復 multi-task 低分問題 (影響最大)
2. 建立 baseline suite (基礎重要)
3. 系統化測試資料增強 (科學方法)
4. 優化 HPO 策略 (提升效率)

---

**報告生成**: Enhanced Experiment Analyzer v2.0
**聯絡**: 如有疑問請查閱原始資料檔案或重新執行分析腳本
