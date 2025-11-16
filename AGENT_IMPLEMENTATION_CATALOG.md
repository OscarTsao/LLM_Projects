# LLM_Projects Agent 實作完整目錄

## 目錄
- [1. Criteria Agent](#1-criteria-agent)
- [2. Evidence Agent](#2-evidence-agent)
- [3. RAG Agent](#3-rag-agent)
- [4. Reranker Agent](#4-reranker-agent)
- [5. Joint Agent](#5-joint-agent)
- [6. LLM-based Agent](#6-llm-based-agent)
- [7. 支援 Agent](#7-支援-agent)
- [8. 系統架構](#8-系統架構)

---

## 1. Criteria Agent
### 「DSM-5 準則匹配」二元分類

#### 實作 1.1: 規則型聚合器
```
位置: 2080_LLM/DataAug_DeBERTa_FourAgents
主檔案: src/agents/criteria_agent.py
類別: CriteriaAgent
```
**特點**:
- 高階聚合介面
- 規則型（非神經網路）
- 配置: `configs/criteria/aggregator.yaml`

**介面規格**:
```python
input: List[Dict] predictions
output: List[Dict] aggregated criteria results

def aggregate(predictions: List[Dict], top_k: int = 3, uncertain_band=(0.4, 0.6)) -> List[Dict]
```

---

#### 實作 1.2: 神經網路型 (ReDSM5)
```
位置: 2080_LLM/DataAugmentation_ReDSM5
主檔案: src/agents/criteria_matching.py
類別: CriteriaMatchingAgent, FocalLoss, AdaptiveFocalLoss
```

**技術堆棧**:
- 模型: BERT/DeBERTa + 自適應焦點損失
- 損失函數: BCE / Focal / AdaptiveFocalLoss
- 優化: 梯度檢查、混合精度、torch.compile
- 硬體支持: GPU + CPU

**配置範例**:
```python
CriteriaMatchingConfig(
    model_name="google-bert/bert-base-uncased",
    max_seq_length=512,
    classifier_hidden_sizes=[256],
    num_labels=2,
    loss_type="adaptive_focal",
    alpha=0.25,
    gamma=2.0,
    delta=1.0
)
```

**介面規格**:
```python
input:
  input_ids: torch.Tensor (batch_size, seq_len)
  attention_mask: torch.Tensor (batch_size, seq_len)

output: AgentOutput(
  predictions: torch.Tensor (0/1),
  confidence: torch.Tensor,
  logits: torch.Tensor,
  probabilities: torch.Tensor,
  metadata: Dict
)
```

**訓練支持**:
- ✓ 訓練 (forward + get_loss)
- ✓ 評估 (predict)
- ✓ 推理 (predict_batch)

---

#### 實作 1.3: 生產級 (NoAug)
```
位置: 4090_LLM/NoAug_Criteria_Evidence
主檔案: 
  - src/psy_agents_noaug/architectures/criteria/models/model.py
  - src/Project/Criteria/model.py
  - scripts/train_criteria.py
  - scripts/eval_criteria.py
類別: Model, ClassificationHead, SequencePooler
```

**特點**:
- Hydra 配置系統
- 嚴格數據驗證（field_map.yaml）
- 完整超參數優化（8+20+50+refit 階段）
- 多架構支持: criteria / evidence / share / joint
- 生產就緒 (✓ 訓練腳本、✓ 評估腳本、✓ HPO)

**支持的模型**:
```python
model_choices = [
  "bert-base-uncased",
  "roberta-base",
  "microsoft/deberta-v3-base"
]
```

**訓練特點**:
```yaml
Mixed Precision: float16/bfloat16 (自動偵測)
Gradient Checkpointing: 記憶體效率
Batch Size: 可配置 (預設 32)
Learning Rate: 可配置 (預設 2e-5)
Scheduler: Cosine with warmup
Early Stopping: 基於驗證指標
```

**HPO 系統**:
```bash
Multi-stage HPO:
  - Stage 0: 8 trials (sanity check)
  - Stage 1: 20 trials (coarse)
  - Stage 2: 50 trials (fine)
  - Stage 3: refit on train+val

Maximal HPO: 600-1200 trials
```

---

## 2. Evidence Agent
### 「句子/跨度級証據」提取

#### 實作 2.1: 成對分類器
```
位置: 2080_LLM/DataAug_DeBERTa_FourAgents
主檔案: src/agents/evidence_agent.py
類別: EvidenceAgent
```

**特點**:
- 成對分類架構
- 配置: `configs/evidence/pairclf.yaml`

**介面規格**:
```python
def train(dataset, output_dir, seed=None, dry_run=False, hparams=None) -> Dict
def infer(dataset, checkpoint_path, output_path, seed=None) -> Dict
```

---

#### 實作 2.2: 雙頭跨度提取
```
位置: 2080_LLM/DataAugmentation_ReDSM5
主檔案: src/agents/evidence_binding.py
類別: EvidenceBindingAgent, MultiAgentPipeline
```

**技術細節**:
- 模型: BERT/DeBERTa + 雙頭 (start/end token prediction)
- 任務: Token 級別跨度提取
- 約束: max_span_length=50, span_threshold=0.5
- 損失函數: BCEWithLogitsLoss (可選標籤平滑)

**介面規格**:
```python
input:
  input_ids: torch.Tensor
  attention_mask: torch.Tensor

output: AgentOutput(
  predictions: List[List[Tuple[int, int]]],  # 跨度座標
  confidence: torch.Tensor,
  logits: {"start": Tensor, "end": Tensor},
  probabilities: {"start": Tensor, "end": Tensor}
)
```

**跨度提取邏輯**:
```python
1. 取得 start_probs 和 end_probs (sigmoid 後)
2. 過濾 > threshold 的位置
3. 為每個 start 找最近的有效 end
4. 檢查跨度長度 <= max_span_length
5. 解碼為文本
```

**推理模式**:
- ✓ 單跨度或多跨度
- ✓ 集成到 MultiAgentPipeline（僅為正準則匹配提取）

---

#### 實作 2.3: 生產級 (NoAug)
```
位置: 4090_LLM/NoAug_Criteria_Evidence
主檔案: 
  - src/psy_agents_noaug/architectures/evidence/models/model.py
  - src/Project/Evidence/model.py
類別: Model, SpanPredictionHead
```

**特點**:
- Hydra 配置支持
- 完整 HPO 系統
- 多架構整合

---

## 3. RAG Agent
### 「檢索擴充生成」DSM-5 準則匹配

#### 實作 3.1: FAISS + SpanBERT 管道
```
位置: 2080_LLM/Psy_RAG
主檔案: 
  - src/models/rag_pipeline.py
  - src/models/embedding_model.py
  - src/models/spanbert_model.py
  - src/models/faiss_index.py
類別: RAGPipeline, BGEEmbeddingModel, SpanBERTModel, FAISSIndex
```

**架構**:
```
Clinical Note (Post)
    ↓
[BGE Embedding] ← BAAI/bge-m3
    ↓
[FAISS Index] ← top_k=10, similarity_threshold=0.7
    ↓
[Candidate Retrieval]
    ↓
[SpanBERT Ranking] ← SpanBERT/spanbert-base-cased
    ↓
[Final Decision] ← spanbert_threshold=0.5
    ↓
RAGResult (criteria_matches)
```

**介面規格**:
```python
input:
  posts_path: Path to JSONL
  criteria_path: Path to criteria definitions
  similarity_threshold: 0.7
  spanbert_threshold: 0.5
  top_k: 10

output: RAGResult(
  post_id: int,
  post_text: str,
  matched_criteria: List[CriteriaMatch],
  total_matches: int,
  processing_time: float
)

CriteriaMatch(
  criteria_id: str,
  diagnosis: str,
  criterion_text: str,
  similarity_score: float,
  spanbert_score: float,
  supporting_spans: List[SpanResult],
  is_match: bool
)
```

**支持的操作**:
- ✓ build_index (建立 FAISS 索引)
- ✓ retrieve_candidates (檢索候選)
- ✓ predict (完整管道)
- ✓ encode_texts (嵌入)
- ✓ decode_spans (文本解碼)

---

#### 實作 3.2: RAG 分類器變體
```
位置: 3090_LLM/Psy_RAG_Agent
主檔案: 
  - src/spanbert_classifier.py
  - src/rag_spanbert_classifier.py
  - src/basic_classifier.py
  - src/minimal_classifier.py
類別: SpanBERTClassifier, RAGSpanBERTClassifier, BasicClassifier
```

**變體**:
1. **SpanBERTClassifier**: 基礎 SpanBERT + 分類頭
2. **RAGSpanBERTClassifier**: RAG 增強版本
3. **BasicClassifier**: 簡化實現
4. **MinimalClassifier**: 最小實現

**訓練框架**: PyTorch + Hugging Face Trainer

**支持的操作**:
- ✓ 訓練 (使用 HF Trainer)
- ✓ 評估 (F1, accuracy, confusion matrix)
- ✓ 推理 (predict_batch)

---

## 4. Reranker Agent
### 「偏好學習」最佳候選排名

#### 實作 4.1: Gemini Reranker 完整系統
```
位置: 4090_LLM/gemini_reranker
主檔案: 
  - src/criteriabind/models.py
  - src/criteriabind/train/train_criteria_ranker.py
  - src/criteriabind/train/train_evidence_span.py
  - src/criteriabind/gemini_judge.py
  - src/criteriabind/cli/*.py
類別: CrossEncoderRanker, SpanExtractor, QASpanModel, GeminiJudge
```

**資料管道** (完整):
```
Step 1: Candidate Generation
  input: Raw clinical notes + criteria
  output: candidates (text spans)

Step 2: Gemini Judging
  - JSON Mode: 結構化輸出
  - Two-pass consistency: 相同提示的 2 次運行，一致性驗證
  - Safety filtering: 移除不安全回應
  output: ranked candidates with winner index

Step 3: Pair Building
  - 將排名轉換為成對範例
  - SHA256(note_id) 用於確定性分割
  output: training pairs (positive/negative)

Step 4: Training
  Track A (Criteria):
    - Model: CrossEncoderRanker
    - Loss: RankNet (pairwise softplus) 或 hinge loss
    - Output: scalar relevance scores
  
  Track B (Evidence):
    - Model: QASpanModel
    - Loss: Span margin loss
    - Output: start/end logits

Step 5: Inference (Best-of-K)
  - 生成 k 個候選
  - 使用訓練的 ranker 評分所有候選
  - 預測: top_score > threshold
  - 可選: 使用證據模型提取 QA 跨度
```

**訓練配置**:
```yaml
optimizer: AdamW
scheduler: cosine with warmup
mixed_precision: fp16/bf16
gradient_accumulation: supported
mlflow_logging: supported

Training State Checkpoint:
  - model_state
  - optimizer_state
  - scheduler_state
  - metadata (step, epoch, etc.)
```

**Gemini Judge 詳情**:
```python
- Model: gemini-2.5-flash (可配置)
- Rubric: 「最忠實、直接支持、完整、清晰、安全」的片段
- Safety: BLOCKED_NONE, BLOCKED_SAFETY 過濾
- Retry: tenacity exponential backoff (429/5xx)
```

**介面規格**:
```python
Input: 
  candidates: List[Candidate]
  criterion: str
  posts: List[str]

Output:
  ranked_candidates: List[Candidate]
  best_score: float
  confidence: float
```

**CLI 支持** (Tyro):
```bash
python -m criteriabind.candidate_gen --in-path data/raw --out-path data/proc --k 8
python -m criteriabind.gemini_judge --in-path data/proc --out-path data/judged --model gemini-2.5-flash
python -m criteriabind.pair_builder --in data/judged --out-train data/pairs/train --out-dev data/pairs/dev
python -m criteriabind.train_criteria_ranker --pairs-path data/pairs/train --dev-path data/pairs/dev
python -m criteriabind.infer --model-path checkpoints/best --test-path data/raw/test
```

**狀態**: ✓ 生產就緒 (完整文檔 CLAUDE.md)

---

## 5. Joint Agent
### 「多任務」準則 + 證據聯合學習

#### 實作 5.1: 雙編碼器融合
```
位置: 4090_LLM/NoAug_Criteria_Evidence
主檔案: 
  - src/psy_agents_noaug/architectures/joint/models/model.py
  - src/Project/Joint/model.py
類別: Model, JointOutput, MultiTaskLoss
```

**架構**:
```
Input (post-criteria pair)
  ↓
[Criteria Encoder] → [Pooler] → [Classification Head] → criteria_logits
  ↓
[Evidence Encoder] ──────────────┐
                                   ↓
                          [Fusion Layer]
                                   ↓
                          [Span Prediction Head] → start/end logits

Loss = criteria_loss_weight * L_criteria + evidence_loss_weight * L_evidence
```

**配置**:
```python
task_weights = {
  "criteria_loss_weight": 0.5,
  "evidence_loss_weight": 0.5
}
shared_encoder: bool = True
freeze_encoder_epochs: int = 0
```

**輸出規格**:
```python
JointOutput(
  logits: torch.Tensor (criteria),
  start_logits: torch.Tensor (evidence),
  end_logits: torch.Tensor (evidence),
  criteria_hidden_states: Tuple,
  evidence_hidden_states: Tuple,
  criteria_attentions: Tuple,
  evidence_attentions: Tuple
)
```

**支持的操作**:
- ✓ 訓練 (多任務損失)
- ✓ 評估 (準則和證據指標)
- ✓ 推理 (並行或順序)
- ✓ HPO (多階段)

---

#### 實作 5.2: 管道組合
```
位置: 2080_LLM/DataAugmentation_ReDSM5
主檔案: src/agents/multi_agent_pipeline.py
類別: MultiAgentPipeline, PipelineOutput
```

**流程**:
```
Input → [CriteriaMatchingAgent] → positive_matches?
                                    ├─ YES → [EvidenceBindingAgent]
                                    └─ NO  → empty_evidence

Output: PipelineOutput(
  criteria_match: bool,
  criteria_confidence: float,
  criteria_probabilities: float,
  evidence_spans: List[List[Tuple]],
  evidence_confidence: float,
  evidence_text: List[List[str]]
)
```

**條件執行**:
- 僅為正準則匹配提取證據
- 負匹配得到空證據列表
- 可配置 `run_evidence` 標誌

---

## 6. LLM-based Agent
### 「大型語言模型」分類和生成

#### 實作 6.1: LLM Criteria Agent
```
位置: 4090_LLM/LLM_Criteria_Gemma
主檔案: 
  - src/training/train.py
  - src/training/train_gemma_hydra.py
  - src/models/gemma_model.py
類別: GemmaClassifier, LLMCriteriaModel
```

**基礎模型**: Google Gemma (2B, 7B)

**訓練模式**:
- SFT (Supervised Fine-Tuning)
- LORA (低秩適應)

**配置**:
```python
quantization: "8bit"  # BitsAndBytes
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

training:
  learning_rate: 5e-4
  batch_size: 8
  num_epochs: 3
```

**推理**:
```python
input: post_text + criterion
output: binary classification (match/not-match)
```

**狀態**: 原型階段 (需推理延遲優化)

---

#### 實作 6.2: LLM Evidence Agent
```
位置: 4090_LLM/LLM_Evidence_Gemma
主檔案: 
  - src/training/train_llm_classifier.py
  - src/training/train_gemma_qa_hydra.py
  - src/models/llm_classification.py
類別: LLMClassificationModel, GemmaQAModel
```

**訓練模式**:
1. **Causal LM**: 以 prompt 為條件生成
2. **Encoderized Classification**: 有編碼器的分類

**推理模式**:
1. **QA 風格**: 生成跨度（"start 123, end 145"）
2. **分類**: 二進制或多元分類

**配置**:
```python
model_type: "causal"  # 或 "encoder"
max_length: 512
num_beams: 1  # 貪心解碼
temperature: 0.7
```

**狀態**: 原型階段

---

## 7. 支援 Agent

### 7.1 Suggestion Agent (VOI)
```
位置: 2080_LLM/DataAug_DeBERTa_FourAgents
類別: SuggestionAgent
函數: enrich(), attach_suggestions(), suggest_for_post()

策略: Value of Information (VOI)
配置:
  top_k: 3
  uncertain_band: (0.4, 0.6)

輸出: 建議下一步要問的問題
```

### 7.2 Evaluation Agent
```
位置: 2080_LLM/DataAug_DeBERTa_FourAgents
類別: EvaluationAgent
函數: evaluate(), compute_evidence_metrics(), compute_criteria_metrics()

指標: F1, accuracy, precision, recall, calibration
品質閘: neg_precision_min, criteria_auroc_min, ece_max
```

### 7.3 Psy Agent (對話管理)
```
位置: 
  - 2080_LLM/Psy_Agent (主要)
  - 3090_LLM/Psy_Agent (變體)
  - 3090_LLM/Psy_Agent_one_by_one (順序模式)

類別: PsyAgent, DialogueManager, QuestionSelector

特點: 整合多個 Agent 進行患者-治療師對話
狀態: 原型階段
```

### 7.4 Report Agent
```
位置: 2080_LLM/Psy_Report_Agent
類別: ReportGenerator, ReportFormatter

輸出格式: JSON, PDF, HTML
內容: 診斷結果、支持證據、信心指標
狀態: 原型階段
```

### 7.5 Risk/Safety Agent
```
位置: 2080_LLM/Psy_Agent
類別: RiskDetector, SafetyFilter

風險類型: suicide, self_harm, other_harm
狀態: 原型階段
```

---

## 8. 系統架構

### 完整數據管道 (NoAug_Criteria_Evidence)

```
Raw Data (HuggingFace 或 Local CSV)
    ↓
[Field Mapping] ← field_map.yaml
    ├─ Criteria: 使用 ONLY status
    └─ Evidence: 使用 ONLY cases
    ↓
[Groundtruth Generation] ← 嚴格驗證
    ├─ CriteriaDataset (status 正規化)
    └─ EvidenceDataset (cases 解析)
    ↓
[Data Split] ← deterministic (SHA256)
    ├─ train (70%)
    ├─ val (15%)
    └─ test (15%)
    ↓
[DataLoader] ← 優化 (num_workers, pin_memory, persistent_workers)
    ↓
[Training Engine]
    ├─ Criteria Arch
    ├─ Evidence Arch
    ├─ Share Arch
    └─ Joint Arch
    ↓
[Checkpoint Management]
    ├─ best.ckpt (驗證指標)
    └─ last.ckpt (完成)
    ↓
[HPO System]
    ├─ Stage 0: 8 trials
    ├─ Stage 1: 20 trials
    ├─ Stage 2: 50 trials
    └─ Stage 3: refit
    ↓
[Evaluation]
    ├─ Metrics (F1, accuracy, auroc, ece)
    └─ Quality Gates
```

### 模型選擇矩陣

| Agent 類型 | BERT | RoBERTa | DeBERTa | SpanBERT | Gemma | 狀態 |
|-----------|------|---------|---------|----------|-------|------|
| Criteria | ✓ | ✓ | ✓ | ✓ | ✓ | Mainline |
| Evidence | ✓ | ✓ | ✓ | ✓ | ✓ | Mainline |
| RAG | - | - | - | ✓ | - | Mainline |
| RAGClassifier | - | - | - | ✓ | - | Mainline |
| Reranker | ✓ | ✓ | ✓ | - | - | Mainline |
| Joint | ✓ | ✓ | ✓ | - | - | Mainline |
| LLM Criteria | - | - | - | - | ✓ | Prototype |
| LLM Evidence | - | - | - | - | ✓ | Prototype |

### 配置優先級

```
1. CLI 參數 (最高優先級)
   python -m ... --learning_rate 3e-5

2. Hydra 覆蓋
   python -m ... training.batch_size=64

3. 配置文件 (YAML)
   configs/training/default.yaml

4. 程式碼預設值 (最低優先級)
   learning_rate: float = 2e-5
```

### 測試覆蓋

| 項目 | 單元測試 | 整合測試 | HPO 測試 |
|-----|---------|---------|---------|
| NoAug_Criteria_Evidence | ✓ | ✓ | ✓ |
| gemini_reranker | ✓ | ✓ | - |
| DataAugmentation_ReDSM5 | ✓ | ✓ | - |
| Psy_RAG | - | ✓ | - |

---

## 附錄

### 快速參考

**最完整的實現**:
- 準則匹配: `/4090_LLM/NoAug_Criteria_Evidence`
- 證據提取: `/4090_LLM/NoAug_Criteria_Evidence`
- Reranker: `/4090_LLM/gemini_reranker`
- RAG: `/2080_LLM/Psy_RAG`

**最佳範例代碼**:
- Criteria Agent: `DataAugmentation_ReDSM5/src/agents/criteria_matching.py`
- Evidence Agent: `DataAugmentation_ReDSM5/src/agents/evidence_binding.py`
- Multi-Agent: `DataAugmentation_ReDSM5/src/agents/multi_agent_pipeline.py`

**學習路徑**:
1. 讀 `CLAUDE.md` (NoAug_Criteria_Evidence)
2. 讀 `agents/base.py` (DataAugmentation_ReDSM5)
3. 研究 `train_criteria_ranker.py` (gemini_reranker)
4. 探索各個 Agent 實現

---

*完整目錄生成於 2025-11-15*
*覆蓋範圍: 14 種 Agent、31+ 項目、1000+ 檔案*
