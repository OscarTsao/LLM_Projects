# Research & Design Decisions - Addendum

**Feature**: Storage-Optimized Training & HPO Pipeline (Extended Configuration)
**Date**: 2025-10-10
**Phase**: 0 Addendum (Expanded HPO Search Space)

## Overview

This addendum extends the original research.md with significantly expanded HPO search space including:
- Parameter-Efficient Fine-Tuning (PEFT) methods (LoRA, Adapters, IA³)
- Extended optimizer suite (10 optimizers)
- Comprehensive scheduler options (10 schedulers)
- Domain Adaptive Pretraining (DAPT) and Task Adaptive Pretraining (TAPT)
- Hardware optimization for RTX 3090/5090 (24GB VRAM)
- Dynamic batch size discovery
- Early stopping with extended training budget (max 100 epochs)

---

## 19. Extended Optimizer Suite

### Decision: 10 Optimizers with Adaptive Variants

**Optimizers to Implement**:

1. **AdamW** (`torch.optim.AdamW`):
   - Weight decay fix for Adam
   - Default choice for transformers
   - HPO params: `lr`, `weight_decay`, `betas=(β1, β2)`, `eps`

2. **Adam** (`torch.optim.Adam`):
   - Classic adaptive LR optimizer
   - Baseline comparison

3. **Adafactor** (`transformers.optimization.Adafactor`):
   - Memory-efficient (no momentum buffers for large models)
   - Scale-invariant learning rate
   - HPO params: `scale_parameter`, `relative_step`, `warmup_init`

4. **RAdam** (`torch.optim.RAdam` or custom):
   - Rectified Adam, addresses convergence issues
   - Automatic warmup adjustment

5. **LAMB** (Large Batch Optimization for BERT):
   - Layer-wise adaptive moments
   - Enables large batch sizes without LR tuning
   - HPO params: `lr`, `weight_decay`, `betas`

6. **SGD** (`torch.optim.SGD`):
   - Baseline stochastic gradient descent
   - HPO params: `lr`, `momentum=0`, `weight_decay`

7. **SGD + Momentum** (`torch.optim.SGD`):
   - Classical momentum-based optimization
   - HPO params: `lr`, `momentum=0.9`, `nesterov={True,False}`

8. **NovoGrad**:
   - Normalized gradients per layer
   - Memory-efficient (no per-parameter momentum)
   - HPO params: `lr`, `betas`, `weight_decay`

9. **AdaBelief** (custom or `adabelief_pytorch`):
   - Adapts step size based on belief in gradient direction
   - Claims faster convergence than Adam
   - HPO params: `lr`, `betas`, `eps`, `weight_decay`

10. **Lion** (`lion_pytorch` or implementation):
    - Sign-based optimizer (memory-efficient)
    - Competitive with AdamW on LLMs
    - HPO params: `lr`, `betas`, `weight_decay`

**HPO Configuration**:
```yaml
optimizer:
  type: categorical([adamw, adam, adafactor, radam, lamb, sgd, sgd_momentum, novograd, adabelief, lion])
  lr: loguniform(1e-6, 5e-5)  # Encoder LR
  head_lr: loguniform(5e-6, 2e-4)  # Task head LR
  weight_decay: uniform(0.0, 0.1)
  betas:
    categorical([(0.9, 0.999), (0.9, 0.98), (0.95, 0.999)])
  eps: categorical([1e-8, 1e-7, 1e-6])
```

**Rationale**:
- AdamW dominant in transformers, but alternatives may suit specific models
- LAMB enables large-batch training (memory-constrained GPUs)
- Adafactor reduces memory for billion-parameter models
- Lion offers efficiency with competitive performance

---

## 20. Comprehensive Scheduler Suite

### Decision: 10 LR Schedulers with Warmup Strategies

**Schedulers to Implement**:

1. **None** (Constant LR):
   - No scheduling, fixed LR throughout training

2. **Linear** (`transformers.get_linear_schedule_with_warmup`):
   - Linear decay from peak LR to 0
   - Standard for BERT-style training

3. **Cosine** (`transformers.get_cosine_schedule_with_warmup`):
   - Cosine annealing from peak to min LR
   - Smooth decay, popular for vision + NLP

4. **Cosine with Restarts** (`transformers.get_cosine_with_hard_restarts_schedule_with_warmup`):
   - Periodic LR resets to escape local minima
   - HPO param: `num_cycles` or `T_mult`

5. **Polynomial** (`transformers.get_polynomial_decay_schedule_with_warmup`):
   - Polynomial decay (power parameter HPO-tunable)
   - HPO param: `power` (e.g., 1.0 = linear, 2.0 = quadratic)

6. **Constant** (no warmup, no decay):
   - Fixed LR, no warmup

7. **Constant with Warmup** (`transformers.get_constant_schedule_with_warmup`):
   - Warmup then constant LR

8. **Exponential** (`torch.optim.lr_scheduler.ExponentialLR`):
   - Exponential decay: `lr = lr_0 * gamma^epoch`
   - HPO param: `gamma` (e.g., 0.95-0.99)

9. **OneCycleLR** (`torch.optim.lr_scheduler.OneCycleLR`):
   - Cycles LR from low → high → low (one cycle per training)
   - HPO params: `max_lr`, `pct_start` (warmup fraction), `div_factor`, `final_div_factor`

10. **ReduceLROnPlateau** (`torch.optim.lr_scheduler.ReduceLROnPlateau`):
    - Reduce LR when validation metric plateaus
    - HPO params: `patience`, `factor`, `threshold`

**Warmup Strategies**:

1. **Warmup Ratio**:
   - Linear warmup for first `warmup_ratio * total_steps` steps
   - HPO range: `[0.0, 0.06, 0.1, 0.2]`

2. **Warmup Steps**:
   - Linear warmup for fixed number of steps
   - HPO range: `[0, 500, 1000, 2000]`

3. **No Warmup**:
   - Start at peak LR immediately

**HPO Configuration**:
```yaml
scheduler:
  type: categorical([none, linear, cosine, cosine_restart, polynomial, constant, constant_warmup, exponential, one_cycle, reduce_plateau])
  warmup_strategy: categorical([ratio, steps, none])
  warmup_ratio: uniform(0.0, 0.2)  # If warmup_strategy=ratio
  warmup_steps: categorical([0, 500, 1000, 2000])  # If warmup_strategy=steps
  # Scheduler-specific params
  cosine_num_cycles: categorical([1, 2, 4])  # For cosine_restart
  cosine_T_mult: categorical([1, 2])
  polynomial_power: uniform(1.0, 3.0)
  exponential_gamma: uniform(0.95, 0.99)
  one_cycle_pct_start: uniform(0.1, 0.3)
  reduce_plateau_patience: categorical([2, 5, 10])
  reduce_plateau_factor: uniform(0.5, 0.9)
```

**Rationale**:
- Linear/cosine schedulers standard for transformers
- OneCycleLR effective for limited-epoch budgets
- ReduceLROnPlateau adaptive to validation performance
- Cosine with restarts helps avoid local minima in long training

---

## 21. Parameter-Efficient Fine-Tuning (PEFT)

### Decision: Support 8 PEFT Methods via Hugging Face PEFT Library

**PEFT Methods**:

1. **None** (Full Fine-Tuning):
   - All parameters trainable
   - Baseline comparison

2. **LoRA** (Low-Rank Adaptation):
   - Injects trainable low-rank matrices into attention layers
   - Params: `r` (rank), `alpha` (scaling), `target_modules`, `dropout`
   - Memory: ~0.1-1% of full model parameters
   - HPO: `r ∈ [4, 8, 16, 32]`, `alpha ∈ [8, 16, 32, 64]`

3. **LoRA+** (LoRA with optimized learning rates):
   - Separate LRs for low-rank matrices A and B
   - Often converges faster than standard LoRA

4. **AdaLoRA** (Adaptive LoRA):
   - Dynamically allocates rank budget across layers
   - Prunes less important adapters during training
   - HPO: Initial rank budget, target rank

5. **Pfeiffer Adapter** (Bottleneck adapter after FFN):
   - Adds bottleneck layers after each transformer FFN
   - HPO: `bottleneck_dim ∈ [32, 64, 128, 192]`, `adapter_dropout`

6. **Houlsby Adapter** (Parallel adapter):
   - Adds adapters in parallel to attention and FFN
   - Higher capacity than Pfeiffer but more parameters

7. **Compacter** (Compact adapter with low-rank + shared parameters):
   - Parameter sharing across layers via hypercomplex multiplication
   - More parameter-efficient than standard adapters

8. **IA³** (Infused Adapter by Inhibiting and Amplifying Inner Activations):
   - Rescales activations via learned vectors (very few params)
   - No additional layers, just element-wise scaling
   - HPO: `enable_ffn ∈ [true, false]` (apply to FFN in addition to attention)

**PEFT Target Modules** (HPO-searchable):
```yaml
peft.target_modules:
  - [query, value]  # Minimal: Only Q and V projections
  - [query, key, value, output]  # Full attention
  - [query, value, ffn.up, ffn.down]  # Attention + FFN
  - [query, key, value, output, ffn.up, ffn.down]  # All projections
```

**Freezing Strategies** (HPO-searchable):
```yaml
freeze_strategy:
  - none: All encoder layers unfrozen (full fine-tune or PEFT on all layers)
  - freeze_layers_n: Freeze bottom N layers
    - n ∈ [0, 2, 6]  # 0=unfreeze all, 2=freeze bottom 2, 6=freeze bottom half
  - freeze_encoder: Freeze entire encoder, train heads only
  - adapter: Use PEFT method (encoder frozen except adapters/LoRA)
```

**Layer-Specific Freezing Options**:
```yaml
unfreeze_layernorm: [true, false]  # Unfreeze LayerNorm even if encoder frozen
unfreeze_pooler: [true, false]  # Unfreeze pooling layer
```

**HPO Configuration**:
```yaml
peft:
  method: categorical([none, lora, lora_plus, adalora, pfeiffer, houlsby, compacter, ia3])
  r: categorical([4, 8, 16, 32])  # LoRA rank
  alpha: categorical([8, 16, 32, 64])  # LoRA alpha
  dropout: uniform(0.0, 0.2)
  bias: categorical([none, lora_only, all])
  target_modules: categorical([...])  # As above
  adapter_bottleneck: categorical([32, 64, 128, 192])  # For Pfeiffer/Houlsby
  adapter_dropout: uniform(0.1, 0.3)
  adapter_layers: categorical([all, top4, top8])  # Which layers get adapters
  ia3_enable_ffn: categorical([true, false])
  peft_lr: loguniform(1e-4, 1e-3)  # Separate LR for PEFT params
  freeze_layers_n: categorical([0, 2, 6])
  unfreeze_layernorm: categorical([true, false])
  unfreeze_pooler: categorical([true, false])
```

**Rationale**:
- PEFT drastically reduces trainable parameters (0.1-10% of full model)
- Enables training large models (e.g., DeBERTa-large) on limited VRAM
- LoRA most popular, but adapters and IA³ offer alternatives
- Freezing bottom layers preserves general knowledge, adapts top layers to task

**Implementation**:
- Use Hugging Face `peft` library (`pip install peft`)
- Wrap base model: `model = get_peft_model(base_model, peft_config)`
- Save only PEFT weights (<<100MB vs multi-GB full model)

---

## 22. Domain/Task Adaptive Pretraining (DAPT/TAPT)

### Decision: Optional Continued Pretraining Phase

**DAPT (Domain Adaptive Pretraining)**:
- Continue masked language modeling (MLM) on mental health text corpus before task fine-tuning
- Adapts general-purpose model (e.g., BERT) to mental health domain
- HPO params:
  - `use_dapt ∈ [true, false]`
  - `dapt_epochs ∈ [0, 1, 2, 3]`
  - `dapt_mlm_prob ∈ uniform(0.15, 0.3)` (mask probability)

**TAPT (Task Adaptive Pretraining)**:
- Continue MLM on RedSM5 training data specifically (task-specific corpus)
- Adapts to vocabulary/patterns of target task
- HPO params:
  - `use_tapt ∈ [true, false]`
  - `tapt_epochs ∈ [0, 1, 2]`
  - `tapt_mlm_prob ∈ uniform(0.15, 0.3)`

**Masking Strategies**:
```yaml
masking_style:
  - token: Random token masking (BERT-style)
  - whole_word: Mask entire words (spans)
  - span: Mask contiguous spans (SpanBERT-style)
```

**Workflow**:
1. Load pretrained model (e.g., `bert-base-uncased`)
2. (Optional) DAPT: Train MLM on mental health corpus for N epochs
3. (Optional) TAPT: Train MLM on RedSM5 training posts for M epochs
4. Fine-tune on criteria matching + evidence binding tasks

**Rationale**:
- Mental health domain has specialized vocabulary (e.g., "rumination", "anhedonia")
- DAPT/TAPT shown to improve performance on domain-specific tasks
- Low cost: 1-3 epochs of MLM pretraining
- HPO explores whether DAPT/TAPT worth the extra compute

**Implementation**:
- Use `transformers.DataCollatorForLanguageModeling` for MLM
- Separate DAPT/TAPT corpus (e.g., scraped mental health forums, RedSM5 train set)

---

## 23. Enhanced Regularization & Training Dynamics

### Decision: Expanded Regularization Suite

**Additional Regularization Techniques**:

1. **Encoder Dropout** (separate from head dropout):
   - Dropout in transformer encoder layers
   - HPO range: `uniform(0.0, 0.2)`

2. **Attention Dropout**:
   - Dropout on attention weights
   - HPO range: `uniform(0.0, 0.2)`

3. **Token Dropout** (drop entire tokens):
   - Randomly mask tokens during training (different from MLM)
   - HPO range: `uniform(0.0, 0.2)`

4. **Gradient Clipping**:
   - Clip gradient norm to prevent exploding gradients
   - HPO options: `[0.0, 1.0, 2.0, 5.0]` (0.0 = no clipping)

5. **Weight Decay** (L2 regularization):
   - Already included, HPO range: `uniform(0.0, 0.1)`

6. **Layer-Wise LR Decay**:
   - Already included, HPO range: `uniform(0.75, 0.95)`

**Joint Training Loss Weighting**:
```yaml
lambda_class: uniform(0.5, 2.0)  # Weight for criteria matching loss
lambda_span: uniform(0.5, 2.0)  # Weight for evidence binding loss
```
- Total loss: `loss = lambda_class * L_criteria + lambda_span * L_evidence`
- Balances task importance during multi-task learning

---

## 24. Hardware Optimization for RTX 3090/5090

### Decision: Single-GPU Optimization with Dynamic Batch Sizing

**Target Hardware**:
- NVIDIA RTX 3090 (24GB VRAM)
- NVIDIA RTX 5090 (24GB VRAM, hypothetical future GPU)
- CPU: 12-core (e.g., Intel Core i7-12700K, AMD Ryzen 9 5900X)

**Optimization Strategies**:

1. **Mixed Precision Training**:
   - FP16 or BF16 (RTX 3090 supports both)
   - Cuts memory usage ~50%, training time ~30%
   - HPO options: `fp_precision ∈ [fp16, bf16, none]`

2. **Gradient Checkpointing**:
   - Recompute activations during backward pass (trade compute for memory)
   - Enables larger models/batch sizes
   - HPO: `gradient_checkpointing ∈ [true, false]`

3. **Gradient Accumulation**:
   - Accumulate gradients over multiple micro-batches
   - Effective batch size = `batch_size * grad_accum`
   - HPO: `grad_accum ∈ [1, 2, 4, 8]`

4. **Dynamic Batch Size Discovery**:
   - **Pre-HPO calibration**: For each model in catalog, find max batch size that fits in 24GB VRAM
   - **Algorithm**:
     ```python
     def find_max_batch_size(model_id, max_length, fp_precision):
         batch_size = 256  # Start optimistically
         while batch_size > 1:
             try:
                 # Attempt forward + backward pass
                 dummy_batch = create_dummy_batch(batch_size, max_length)
                 outputs = model(**dummy_batch)
                 loss = outputs.loss
                 loss.backward()
                 del outputs, loss
                 torch.cuda.empty_cache()
                 return batch_size  # Success!
             except torch.cuda.OutOfMemoryError:
                 batch_size //= 2  # Halve and retry
         return 1  # Fallback
     ```
   - **Update HPO config**: Set `batch_size` upper bound dynamically per model
   - Example: `bert-base` may fit `batch_size=64`, but `longformer-large` only `batch_size=4`

5. **Separate Train/Eval Batch Sizes**:
   - Evaluation doesn't require gradients → larger batch size possible
   - HPO: `train_batch_size` and `eval_batch_size` as separate parameters
   - Example: `train_batch_size=16`, `eval_batch_size=64`

6. **DataLoader Optimization**:
   - `num_workers=12` (match CPU cores for parallel data loading)
   - `pin_memory=True` (faster CPU → GPU transfer)
   - `persistent_workers=True` (avoid worker restart overhead)

7. **CUDA Optimization**:
   ```python
   torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms
   torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32 on Ampere GPUs
   ```

**Memory Budget Estimation**:
- Model weights: 400MB (base) to 2GB (large)
- Optimizer states (AdamW): 2× model size
- Gradients: 1× model size
- Activations: Scales with `batch_size * max_length * hidden_size`
- Total for `bert-base` + batch_size=32 + max_length=512: ~10GB
- Leaves 14GB for larger models or larger batches

**HPO Configuration Update**:
```yaml
# Dynamic batch size upper bound per model (updated during pre-HPO calibration)
batch_size_upper_bounds:
  bert-base-uncased: 64
  deberta-v3-base: 32
  longformer-base-4096: 8
  spanbert-large-cased: 16
  # ... (auto-populated for all 30 models)

# HPO search space
train_batch_size: categorical([4, 8, 16, 32, ...])  # Up to model-specific upper bound
eval_batch_size: categorical([8, 16, 32, 64, 128])  # Higher than train_batch_size
grad_accum: categorical([1, 2, 4, 8])
fp_precision: categorical([fp16, bf16, none])
gradient_checkpointing: categorical([true, false])
```

**Rationale**:
- RTX 3090's 24GB VRAM sufficient for most models with mixed precision + gradient checkpointing
- Dynamic batch size discovery prevents OOM errors during HPO
- Separate train/eval batch sizes maximizes eval throughput
- 12 CPU cores enable efficient parallel data loading

---

## 25. Extended Training Budget & Early Stopping

### Decision: Max 100 Epochs with Early Stopping (20 Patience)

**Training Configuration**:
```yaml
max_epochs: 100  # Upper limit (unlikely to reach with early stopping)
early_stopping:
  patience: 20  # Stop if no improvement for 20 epochs
  min_delta: 0.001  # Minimum change to qualify as improvement
  metric: val_f1_macro  # Optimization metric
  mode: max  # Maximize the metric
```

**Early Stopping Logic**:
1. Track best validation metric across epochs
2. If `current_val_metric <= best_val_metric + min_delta` for 20 consecutive epochs, stop training
3. Restore best checkpoint (not last checkpoint)

**Rationale**:
- Mental health data small (<10GB), models may converge in 10-30 epochs
- 100 epoch budget allows exploration of slower optimizers (e.g., SGD)
- 20 patience generous enough to escape plateaus but prevents runaway training
- Saves compute: Most trials stop <50 epochs

**HPO Impact**:
- Trials with poor hyperparameters pruned early by Optuna's Median Pruner
- Combined with early stopping: Double safeguard against wasted compute

---

## 26. Complete HPO Search Space (Master Configuration)

### Consolidated Search Space Summary

**Dimensions**: ~100 hyperparameters (categorical + continuous)

**Estimated Combinations**: O(10^30) (increased from O(10^15) in base config)

**Key Additions**:
- 10 optimizers (vs 2 originally)
- 10 schedulers (vs 4 originally)
- 8 PEFT methods (new)
- DAPT/TAPT options (new)
- Extended regularization (encoder dropout, attention dropout, token dropout)
- Dynamic batch sizing per model
- Separate train/eval batch sizes
- Hardware-specific optimizations

**Search Space Categories**:

1. **Model Selection** (30 + PEFT combinations = ~240 effective architectures)
2. **Training Dynamics** (optimizer × scheduler × warmup = 10 × 10 × 3 = 300 combinations)
3. **Regularization** (dropout variants, weight decay, gradient clipping, label smoothing)
4. **Task Architecture** (head types, pooling, coupling, loss functions)
5. **Data** (augmentation, input format, max_length, padding)
6. **PEFT Configuration** (method, rank, target modules, freezing strategy)
7. **Pretraining** (DAPT/TAPT epochs, masking style)
8. **Hardware** (batch size, gradient accumulation, mixed precision, gradient checkpointing)

**Optuna Configuration**:
```yaml
study:
  n_trials: 1000
  sampler: tpe  # Tree-structured Parzen Estimator
  pruner: median  # Prune trials performing worse than median at each epoch
  direction: maximize  # Maximize val_f1_macro
  load_if_exists: true  # Resume from existing study DB
```

---

## 27. Implementation Priorities

### Phase 1 (Core Infrastructure):
1. Dynamic batch size discovery script
2. PEFT integration via Hugging Face `peft` library
3. Extended optimizer/scheduler registry
4. Hardware optimization (mixed precision, gradient checkpointing)

### Phase 2 (HPO Integration):
1. Update `search_space.py` with all new hyperparameters
2. Conditional search space logic (e.g., if PEFT, then PEFT params)
3. Pre-HPO calibration: Run batch size discovery for all 30 models
4. Update `config_schema.yaml` with new fields

### Phase 3 (Advanced Features):
1. DAPT/TAPT pretraining pipeline
2. Multi-objective optimization (F1 vs memory vs training time)
3. Hyperparameter importance analysis (Optuna built-in)
4. Best trial ensemble (combine top-5 trials' predictions)

---

## 28. Updated Success Metrics

**Original Success Criteria** (from spec.md):
- SC-001: ≥60% storage reduction
- SC-002: 100% resume success
- SC-003: 100% metrics logged
- SC-004: 100% trials have JSON reports
- SC-005: 100% model loading success
- SC-006: Environment operational in ≤15 min

**Extended Success Criteria**:
- **SC-007**: Dynamic batch size discovery completes for all 30 models in ≤1 hour
- **SC-008**: PEFT methods reduce trainable parameters by ≥90% vs full fine-tuning
- **SC-009**: Mixed precision training achieves ≥30% speedup vs FP32 (on RTX 3090)
- **SC-010**: Early stopping triggers in ≥50% of trials (validates effectiveness)
- **SC-011**: TPE sampler converges to best trial within first 300 trials (validates search efficiency)
- **SC-012**: Median pruner prunes ≥20% of trials by epoch 5 (saves compute)

---

## Summary of Addendum

This addendum expands the HPO search space by ~100x compared to the base configuration:

| Aspect | Base Config | Extended Config |
|--------|-------------|-----------------|
| Optimizers | 2 (Adam, AdamW) | 10 (AdamW, Adam, Adafactor, RAdam, LAMB, SGD, SGD+Momentum, NovoGrad, AdaBelief, Lion) |
| Schedulers | 4 (linear, cosine, constant, one_cycle) | 10 (+ cosine_restart, polynomial, exponential, reduce_plateau, constant_warmup, none) |
| PEFT Methods | None | 8 (LoRA, LoRA+, AdaLoRA, Pfeiffer, Houlsby, Compacter, IA³) |
| Freezing Strategies | Unfreeze all | 4 (none, freeze_layers_n, freeze_encoder, adapter) |
| Pretraining | None | DAPT + TAPT with configurable epochs |
| Batch Size | Fixed [8, 16, 32] | Dynamic per model [4, 8, 16, 32, 64, ...256] |
| Train/Eval Batch | Same | Separate (eval larger) |
| Hardware Optimization | Basic | Mixed precision, gradient checkpointing, TF32, CPU workers=12 |
| Max Epochs | 30 | 100 (with early stopping patience=20) |
| HPO Trials | 1000 | 1000 (but with pruner to save compute) |
| Search Space Size | O(10^15) | O(10^30) |

**Key Takeaways**:
1. PEFT enables training large models (DeBERTa-large, Longformer-large) on 24GB VRAM
2. Dynamic batch sizing prevents OOM errors during HPO
3. Extended optimizer/scheduler suite explores diverse optimization landscapes
4. DAPT/TAPT may improve performance on domain-specific mental health text
5. Early stopping + Optuna pruner prevent wasted compute on unpromising trials
6. Hardware optimization maximizes RTX 3090 utilization

**Next Steps**:
- Update `config_schema.yaml` and `trial_output_schema.json` with new fields
- Implement dynamic batch size discovery script
- Integrate Hugging Face `peft` library
- Extend `search_space.py` with conditional logic for new hyperparameters
