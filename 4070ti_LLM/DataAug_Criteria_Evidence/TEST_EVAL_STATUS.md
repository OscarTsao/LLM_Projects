# WITH-AUG Test Evaluation - Running

**Status:** ✅ ALL 4 EVALUATIONS RUNNING
**Started:** November 7, 2025 14:18-14:19
**Estimated Completion:** 4-6 hours from start

---

## Running Processes

| Architecture | PID   | Status | Config |
|--------------|-------|--------|--------|
| **Criteria** | 11812 | ✅ Running | Val F1: 0.7096, Model: ConvBERT, Aug: Yes |
| **Evidence** | 12360 | ✅ Running | Val F1: 0.7040, Model: DistilBERT, Aug: No |
| **Share**    | 12413 | ✅ Running | Val F1: 0.8585, Model: DeBERTa, Aug: Yes |
| **Joint**    | 12018 | ✅ Running | Val F1: 0.8397, Model: BERT, Aug: Yes |

**GPU Utilization:** 100% (12.4 GB / 24.6 GB)

---

## What's Happening

Each architecture is being retrained with its best hyperparameters from HPO:
- **100 epochs** of training
- Best configs from hpo_best_configs_summary.json
- Train+val data combined for refit
- Evaluation on **validation set** (note: true test set not accessible via this method)

---

## Monitoring Commands

```bash
# Check all running processes
ps aux | grep "simple_test_eval.py" | grep -v grep

# Monitor GPU usage
nvidia-smi -l 5

# Watch Criteria progress
tail -f logs/test_eval_withaug/criteria.log

# Watch Evidence progress
tail -f logs/test_eval_withaug/evidence.log

# Watch Share progress
tail -f logs/test_eval_withaug/share.log

# Watch Joint progress
tail -f logs/test_eval_withaug/joint.log

# Check results (after completion)
ls -lh outputs/test_eval_*/results.json
```

---

## Expected Timeline

| Architecture | Est. Time | Progress |
|--------------|-----------|----------|
| Criteria     | 1-2 hours | 🔄 Training |
| Evidence     | 1-2 hours | 🔄 Training |
| Share        | 1-2 hours | 🔄 Training |
| Joint        | 1-2 hours | 🔄 Training |

**Total:** ~4-6 hours (all running in parallel on single GPU)

---

## Output Locations

**Logs:** `logs/test_eval_withaug/<arch>.log`
**Results:** `outputs/test_eval_<arch>/results.json`

Each results.json will contain:
- `validation_f1_hpo`: Original HPO validation F1
- `validation_f1_refit`: New validation F1 after refit training
- `f1_macro`, `ece`, `logloss`: Detailed metrics
- `runtime_s`: Training time

---

## Next Steps (After Completion)

1. ✅ Collect results from all 4 architectures
2. ✅ Generate comprehensive comparison report
3. ✅ Compare WITH-AUG vs NO-AUG performance
4. ✅ Statistical analysis of results

---

**Created:** November 7, 2025 14:21
**Script:** `scripts/simple_test_eval.py`
