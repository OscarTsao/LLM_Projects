# LLM_Projects Agent 實作分析報告

## 執行摘要

本報告對 LLM_Projects 中所有 Agent 實作進行了深入分析，涵蓋 14 種 Agent 類型、31+ 個項目和超過 1000+ 個 Python 文件。

### 核心發現

#### 1. **Criteria Agent** (5+ 實作)
- **主要實現**：
  - `DataAug_DeBERTa_FourAgents`: 規則型聚合器
  - `DataAugmentation_ReDSM5`: 神經網路型（BERT+Adaptive Focal Loss）
  - `NoAug_Criteria_Evidence`: 完整 Hydra 系統，支持 HPO

- **技術特點**：
  - 自適應焦點損失（Adaptive Focal Loss）用於類別不平衡
  - 支持多種損失函數：BCE、Focal、Adaptive Focal
  - 硬體最佳化：梯度檢查、混合精度、torch.compile
  - 配置系統：Hydra + field_map.yaml 嚴格驗證

#### 2. **Evidence Agent** (4+ 實作)
- **主要實現**：
  - `DataAug_DeBERTa_FourAgents`: 成對分類器
  - `DataAugmentation_ReDSM5`: 雙頭跨度預測（start/end tokens）
  - `NoAug_Criteria_Evidence`: 配合 Joint/Share 架構

- **技術特點**：
  - Token 級別跨度提取（最大跨度 50）
  - 閾值型跨度匹配（0.5）
  - BCEWithLogitsLoss + 可選標籤平滑
  - 集成 MultiAgentPipeline（條件性執行）

#### 3. **RAG Agent** (2 實作 + Classifier 4 變體)
- **主要實現**：
  - `Psy_RAG`: FAISS + SpanBERT 檢索系統
    - 嵌入模型：BAAI/bge-m3
    - 匹配模型：SpanBERT/spanbert-base-cased
    - 檢索閾值：0.7，SpanBERT 閾值：0.5

  - `Psy_RAG_Agent`: SpanBERT 分類器變體
    - 4 種分類器型式：RAG、Basic、Minimal、Simple
    - 支持訓練和評估

#### 4. **Reranker Agent** (gemini_reranker)
- **架構**：偏好學習管道（Preference Learning Pipeline）
- **Two Tracks**：
  - Track A (Criteria): CrossEncoderRanker + RankNet 損失
  - Track B (Evidence): QASpanModel + Span Margin 損失
- **Judge 系統**：Gemini API + JSON 模式、雙通一致性、安全過濾
- **數據管道**：候選生成 → Gemini 判斷 → 成對構建 → 訓練
- **特點**：完整的 CLAUDE.md 文檔，生產就緒

#### 5. **Joint Agent** (2 實作)
- **主要實現**：
  - `NoAug_Criteria_Evidence`: 雙編碼器 + 融合層
  - `DataAugmentation_ReDSM5`: MultiAgentPipeline 組成

- **技術特點**：
  - 多任務學習（criteria_loss_weight: 0.5, evidence_loss_weight: 0.5）
  - 共享或分離編碼器
  - 條件執行：證據只為正匹配提取

#### 6. **LLM-based Agent** (Gemma 變體)
- **LLM Criteria Agent**：
  - 基礎模型：Google Gemma (2B, 7B)
  - 訓練模式：SFT 或 LORA
  - 量化：8-bit BitsAndBytes
  - 配置：Hydra

- **LLM Evidence Agent**：
  - 訓練模式：Causal LM 或 Encoderized 分類
  - 推理模式：QA 風格跨度生成或分類
  - 優化：LORA + 8-bit 量化

#### 7. **其他 Agent 類型**
- **Suggestion Agent**: 基於 Value of Information (VOI)，建議下一個要問的問題
- **Evaluation Agent**: 完整評估、溫度縮放、品質閘檢查
- **Report Agent**: 臨床報告生成（原型階段）
- **Risk/Safety Agent**: 自殺/自傷/他傷風險偵測（原型階段）
- **Patient Graph Agent**: GNN 型患者信息圖（概念級別）
- **Psy Agent**: 對話管理系統（多代理組合）

---

## 設計模式與架構

### 支持的模型
- **編碼器型**：BERT、RoBERTa、DeBERTa、SpanBERT
- **LLM型**：Google Gemma (2B, 7B)

### 配置系統
- **Hydra**：主要配置框架（NaoAug_Criteria_Evidence、gemini_reranker）
- **YAML**：低級配置文件
- **Pydantic**：數據驗證和類型安全
- **Argparse**：簡單腳本

### 硬體最佳化
- **混合精度**：Float16、BFloat16 自動檢測
- **梯度檢查**：記憶體效率
- **torch.compile**：推理加速
- **LORA**：參數高效訓練
- **8-bit 量化**：降低記憶體消耗

### 訓練框架
- **PyTorch**：核心框架
- **Hugging Face Transformers**：模型
- **MLflow**：實驗追蹤（支持的項目）
- **Optuna**：超參數優化

---

## 項目狀態映射

### 生產就緒（Mainline）
- CriteriaAgent (DataAugmentation_ReDSM5, NoAug_Criteria_Evidence)
- EvidenceAgent (DataAugmentation_ReDSM5, NoAug_Criteria_Evidence)
- RAGAgent (Psy_RAG)
- RAGClassifier (Psy_RAG_Agent 變體)
- **RerankerAgent (gemini_reranker)** ← 最完整
- JointAgent (NoAug_Criteria_Evidence)
- SuggestionAgent (DataAug_DeBERTa_FourAgents)
- EvaluationAgent (DataAug_DeBERTa_FourAgents)

### 原型階段（Prototype）
- LLMCriteriaAgent
- LLMEvidenceAgent
- PsyAgent（對話管理）
- ReportAgent
- RiskSafetyAgent
- PatientGraphAgent

### 基線（Baseline）
- Criteria_Baseline_5Fold_NoAug
- Evidence_Baseline_5Fold_NoAug

---

## 關鍵實現細節

### 嚴格數據驗證（NoAug_Criteria_Evidence）
```yaml
# configs/data/field_map.yaml 強制執行：
- Criteria 使用 ONLY status 字段
- Evidence 使用 ONLY cases 字段
- 違反將導致 AssertionError
```

### 多階段超參數優化（NoAug_Criteria_Evidence）
```bash
Stage 0: 健全性檢查（8 次試驗）
Stage 1: 粗調（20 次試驗）
Stage 2: 細調（50 次試驗）
Stage 3: 重新擬合（train+val）

或最大化 HPO：600-1200 次試驗
```

### Gemini Reranker 管道
```
1. Candidate Generation: 從臨床筆記中提取候選跨度
2. Gemini Judging: Gemini API 對候選進行排名（兩通一致性）
3. Pair Building: 將排名轉換為成對訓練示例
4. Training: CrossEncoderRanker (Track A) 或 QASpanModel (Track B)
5. Inference: Best-of-k 決策
```

---

## 文件位置

### 完整分析
- `AGENT_ANALYSIS_COMPLETE.json` - 結構化數據
- `AGENT_ANALYSIS_SUMMARY.md` - 本文檔

### 關鍵 CLAUDE.md 檔案
- `/home/user/LLM_Projects/4090_LLM/NoAug_Criteria_Evidence/CLAUDE.md` - 完整指南
- `/home/user/LLM_Projects/4090_LLM/gemini_reranker/CLAUDE.md` - Reranker 詳情

### 主要項目
- `2080_LLM/DataAug_DeBERTa_FourAgents` - 四代理模式示例
- `2080_LLM/DataAugmentation_ReDSM5` - 完整基礎架構
- `4090_LLM/NoAug_Criteria_Evidence` - 生產級實現
- `4090_LLM/gemini_reranker` - Gemini 集成

---

## 常見 TODOs 和改進機會

1. **架構整合** (NoAug_Criteria_Evidence):
   - 合併 `src/Project/` 和 `src/psy_agents_noaug/architectures/` (估計 2-4 小時)
   - 統一 CLI 接口

2. **LLM Agent 優化**:
   - 推理延遲優化
   - 多令牌預測支持
   - 品質改進

3. **風險/安全 Agent**:
   - 完整 GNN 實現
   - 風險閾值設定
   - 可解釋性增強

4. **對話系統**:
   - 上下文記憶
   - 輪次管理
   - 對話狀態追蹤

5. **報告生成**:
   - HIPAA 合規性
   - 範本系統
   - PDF/HTML 匯出

---

## 統計

| 指標 | 數值 |
|------|------|
| Agent 類型 | 14 |
| 項目總數 | 31+ |
| Python 文件 | 1000+ |
| 主要實現 | 30+ |
| 完整配置檔 | 200+ |
| 生產就緒 | 8+ |
| 原型階段 | 6+ |

---

## 推薦閱讀順序

1. **快速概覽**：本摘要 + AGENT_ANALYSIS_COMPLETE.json
2. **完整實現**：NoAug_Criteria_Evidence CLAUDE.md
3. **Reranker 詳情**：gemini_reranker CLAUDE.md
4. **代碼級詳情**：各個 Agent 檔案（見 JSON）

---

*報告生成時間: 2025-11-15*
*分析範圍: 非常徹底（Very Thorough）*
