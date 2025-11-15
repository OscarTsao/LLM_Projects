# Verification Suite Quick Start

## One-Command Verification

```bash
bash tools/verify/run_all.sh
```

This will:
1. Install test dependencies
2. Run all 23 verification tests
3. Execute micro-benchmarks
4. Generate reports (verification_report.md)

---

## Common Commands

### Run All Tests
```bash
pytest tests/verify/ -v
```

### Run Specific Test Module
```bash
pytest tests/verify/test_03_evidence_only.py -v
```

### Run Without GPU Tests
```bash
pytest tests/verify/ -v -m "not gpu"
```

### Run in Parallel (4 workers)
```bash
pytest tests/verify/ -n 4
```

### Run with Detailed Output
```bash
pytest tests/verify/ -vv --tb=short
```

---

## Test Categories

### Critical Tests (Must Pass)
- `test_01_registry.py` - Method registry validation
- `test_03_evidence_only.py` - Core property: only evidence modified
- `test_04_determinism.py` - Reproducibility with seeds

### Integration Tests
- `test_02_cli_smoke.py` - CLI functional
- `test_08_manifests.py` - Output integrity
- `test_15_all_methods.py` - Method instantiation

### Optional Tests (May Skip)
- `test_11_gpu_cpu_execution.py` - Requires CUDA
- `test_14_linting.py` - Requires ruff

---

## Interpreting Results

### All Pass
```
========================= 23 passed in 45.23s ==========================
```
→ Verification complete, system ready for production

### Some Skip/XFail
```
=============== 20 passed, 3 skipped, 1 xfailed in 38.12s ==============
```
→ Acceptable if GPU tests skipped (no CUDA) or optional deps missing

### Failures
```
===================== 18 passed, 5 failed in 52.45s ====================
```
→ Review test_results.json for details, fix issues before proceeding

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src.augment'"
```bash
# Ensure you're in project root
cd /home/oscartsao/Developer/DataAugmentation_ReDSM5

# Install project in editable mode
pip install -e .
```

### "FileNotFoundError: conf/augment_methods.yaml"
```bash
# Verify file exists
ls -la conf/augment_methods.yaml

# Run from project root
cd /home/oscartsao/Developer/DataAugmentation_ReDSM5
pytest tests/verify/
```

### "No datasets generated" (test skips)
→ Expected if methods unavailable (missing nlpaug/textattack)
→ Install augmentation dependencies: `pip install -r requirements-augment.txt`

### GPU tests fail
→ Expected on CPU-only systems
→ Tests marked with @pytest.mark.xfail when no CUDA

---

## File Locations

| Item | Location |
|------|----------|
| Test modules | `tests/verify/test_*.py` |
| Fixture data | `tests/fixtures/mini_annotations.csv` |
| Utilities | `tests/verify_utils.py` |
| Scripts | `tools/verify/*.sh`, `tools/verify/*.py` |
| Config | `pytest.ini` |
| Reports | `verification_report.md`, `test_results.json` |

---

## Expected Runtime

| Command | Time | Tests |
|---------|------|-------|
| Single test module | ~5-15s | 1-4 tests |
| Full suite (CPU only) | ~5-8 min | 20-21 tests |
| Full suite (with GPU) | ~8-10 min | 23 tests |
| Parallel (4 workers) | ~3-5 min | 23 tests |

---

## Output Files

After running `tools/verify/run_all.sh`:

```
verification_report.md         # Human-readable summary
verification_summary.json      # Machine-readable results
test_results.json             # Detailed pytest output
benchmark_results.json        # Performance metrics
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run verification suite
  run: |
    bash tools/verify/setup_env.sh
    pytest tests/verify/ -v --json-report --json-report-file=test_results.json
```

### Pre-commit Hook
```bash
#!/bin/bash
pytest tests/verify/test_01_registry.py tests/verify/test_13_no_training_code.py -q
```

---

## Getting Help

1. Check test output: `pytest tests/verify/test_XX_*.py -vv`
2. Review fixture data: `cat tests/fixtures/mini_annotations.csv`
3. Inspect implementation: See `VERIFICATION_SUITE_IMPLEMENTATION.md`
4. Validate environment: `python3 -c "import pytest; import pandas; print('OK')"`

---

**Quick Reference Card**

```bash
# Full verification
bash tools/verify/run_all.sh

# Fast check (critical tests only)
pytest tests/verify/test_01_registry.py tests/verify/test_03_evidence_only.py -v

# Benchmark only
python3 tools/verify/bench_small.py

# Generate report from existing results
python3 tools/verify/generate_report.py
```
