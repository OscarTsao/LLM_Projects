# Multi-Stage HPO Limitations and Recommendations

## Critical Issues Discovered

### 1. Incomplete Implementation in `scripts/run_hpo_stage.py`

**Line 136**: Only criteria task is implemented:
```python
if cfg.task.name == "criteria":
    dataset_path = project_root / "data" / "redsm5" / "redsm5_annotations.csv"
    dataset = CriteriaDataset(...)
else:
    raise NotImplementedError(f"Task {cfg.task.name} not implemented yet")
```

**Impact**:
- ✅ `make hpo-s0 HPO_TASK=criteria` works
- ❌ `make hpo-s0 HPO_TASK=evidence` raises NotImplementedError
- ❌ `make hpo-s0 HPO_TASK=share` raises NotImplementedError
- ❌ `make hpo-s0 HPO_TASK=joint` raises NotImplementedError
- ❌ `make full-hpo-all` fails immediately when it tries evidence

### 2. Missing Optuna MLflow Integration

**Installed**: `optuna==4.5.0` (base package only)
**Missing**: `optuna-integration[mlflow]` extras

While not strictly required for basic Optuna functionality, the multi-stage HPO system (`run_hpo_stage.py`) was designed to work with MLflow integration for tracking, which may cause issues.

### 3. Runtime Constraints

Even if fully implemented, the default trial counts are prohibitive for single-GPU environments:

**Multi-Stage HPO** (per architecture):
- Stage 0: 8 trials × 3 epochs = ~30 minutes
- Stage 1: 20 trials × epochs = ~2-4 hours
- Stage 2: 50 trials × epochs = ~5-10 hours
- **Total per architecture**: ~8-15 hours

**Maximal HPO** (per architecture):
- Criteria: 800 trials × 100 epochs = ~100-150 hours
- Evidence: 1200 trials × 100 epochs = ~150-200 hours
- Share: 600 trials × 100 epochs = ~75-100 hours
- Joint: 600 trials × 100 epochs = ~75-100 hours
- **Total for all**: ~400-550 hours (17-23 days on single GPU)

## Why This Happened

The codebase has **TWO HPO systems** that were never fully reconciled:

### System 1: Multi-Stage HPO (INCOMPLETE)
- **Script**: `scripts/run_hpo_stage.py`
- **Framework**: Hydra + OptunaRunner
- **Status**: ❌ Only criteria implemented
- **Used by**: `make hpo-s0`, `make hpo-s1`, `make hpo-s2`, `make refit`

### System 2: Maximal HPO (COMPLETE)
- **Script**: `scripts/tune_max.py`
- **Framework**: Standalone with argparse
- **Status**: ✅ All 4 agents implemented (criteria, evidence, share, joint)
- **Used by**: `make tune-*-max`, `python -m psy_agents_noaug.cli tune`

## Current Working Commands

### ✅ What Works

**Maximal HPO** (all agents implemented):
```bash
# Individual agents
make tune-criteria-max    # 800 trials
make tune-evidence-max    # 1200 trials
make tune-share-max       # 600 trials
make tune-joint-max       # 600 trials

# All agents sequentially
make maximal-hpo-all      # ~19,000 trials total

# Via CLI
python -m psy_agents_noaug.cli tune --agent criteria --study test --n-trials 10
```

**Training**:
```bash
# Direct training
make train TASK=criteria MODEL=roberta_base
python -m psy_agents_noaug.cli train --agent criteria
```

**Standalone scripts**:
```bash
# Criteria only (production-ready)
python scripts/train_criteria.py
python scripts/eval_criteria.py
```

### ❌ What Doesn't Work

**Multi-Stage HPO**:
```bash
make hpo-s0              # Only works for criteria
make hpo-s1              # Only works for criteria
make hpo-s2              # Only works for criteria
make refit               # Only works for criteria
make full-hpo            # Only works for criteria
make full-hpo-all        # Fails on evidence (NotImplementedError)
```

## Recommended Solutions

### Option 1: Use Maximal HPO System (IMMEDIATE)

**Pros**:
- ✅ Already implemented for all 4 agents
- ✅ No code changes needed
- ✅ Works with current CLI

**Cons**:
- ❌ Very long runtime (hundreds of trials)
- ❌ No progressive refinement stages
- ❌ Single large run instead of iterative

**How to use**:
```bash
# Quick test (reduced trials)
python scripts/tune_max.py --agent criteria --study test-run --n-trials 10 --parallel 1

# Production (default trial counts)
make tune-criteria-max    # 800 trials, ~5-8 hours
make tune-evidence-max    # 1200 trials, ~8-12 hours

# All agents (WARNING: 17-23 days on single GPU)
make maximal-hpo-all
```

### Option 2: Implement Multi-Stage for All Agents (FUTURE WORK)

**Required changes**:

1. **Install MLflow integration**:
```bash
poetry add "optuna-integration[mlflow]"
```

2. **Extend `run_hpo_stage.py`** to support all tasks:
```python
# Add support for evidence, share, joint
if cfg.task.name == "criteria":
    dataset = CriteriaDataset(...)
elif cfg.task.name == "evidence":
    dataset = EvidenceDataset(...)  # Need to implement
elif cfg.task.name == "share":
    dataset = ShareDataset(...)     # Need to implement
elif cfg.task.name == "joint":
    dataset = JointDataset(...)     # Need to implement
```

3. **Create dataset loaders** for evidence/share/joint

4. **Test each stage** for each agent

**Estimated effort**: 8-16 hours of development + testing

### Option 3: Remove Multi-Stage HPO (SIMPLIFICATION)

**Rationale**:
- Multi-stage system is incomplete and unmaintained
- Maximal HPO system is complete and working
- Reduces code duplication
- Clearer user experience

**Changes**:
1. Remove `scripts/run_hpo_stage.py`
2. Remove `configs/hpo/stage*.yaml`
3. Update Makefile to remove hpo-s0/s1/s2/refit targets
4. Update documentation to focus on maximal HPO
5. Keep only `tune` command in CLI

**Pros**:
- ✅ Simpler codebase
- ✅ One clear HPO path
- ✅ No confusion about which system to use

**Cons**:
- ❌ Lose progressive refinement concept
- ❌ No quick sanity check option

## Recommended Path Forward

### For Immediate Use:
1. **Use maximal HPO with reduced trials** for testing:
   ```bash
   python scripts/tune_max.py --agent criteria --study quick-test --n-trials 10
   ```

2. **Run production HPO on appropriate hardware** (multi-GPU cluster):
   ```bash
   # On cluster with 4+ GPUs
   make maximal-hpo-all
   ```

### For Long-Term Solution:
1. **Choose**: Either implement multi-stage for all agents (Option 2) OR remove it entirely (Option 3)
2. **Document** the chosen approach clearly in CLAUDE.md
3. **Update** all Makefile targets and docs consistently
4. **Test** the complete workflow end-to-end

## Immediate Action Required

To make HPO functional right now:

### Quick Fix for Single Agent (Criteria):
```bash
# This works NOW
make hpo-s0 HPO_TASK=criteria    # 8 trials, ~30 min
make tune-criteria-max            # 800 trials, ~5-8 hours
```

### Quick Fix for All Agents:
```bash
# Use maximal HPO system (not multi-stage)
python scripts/run_all_hpo.py --mode maximal --n-trials 50 --parallel 1

# Or individual agents with reduced trials
for agent in criteria evidence share joint; do
    python scripts/tune_max.py --agent $agent --study quick-$agent --n-trials 50
done
```

## Summary

**The multi-stage HPO system is NOT production-ready**:
- ❌ Only 1/4 agents implemented
- ❌ Missing dependency (optuna-integration[mlflow])
- ❌ Runtime would be days even if complete

**The maximal HPO system IS production-ready**:
- ✅ All 4 agents implemented
- ✅ Used successfully in recent commits
- ✅ No missing dependencies
- ⚠️ Still requires significant GPU time (reduce trials for testing)

**Recommendation**: Use maximal HPO system with reduced trials for development/testing, schedule full runs on appropriate hardware.
