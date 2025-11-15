# Quality Gates Documentation

**Project:** PSY Agents - Clinical Text Analysis (NO-AUG → AUG Transformation)
**Version:** 1.0
**Last Updated:** 2025-01-25

---

## Overview

This document defines **10 critical quality gates** that MUST be satisfied before the augmented (AUG) version can be deployed to production.

### Gate Enforcement

**MANDATORY:** All gates must show **PASS** status before production deployment.

### Quality Gate Status Levels

| Status | Symbol | Meaning |
|--------|--------|---------|
| **PASS** | ✅ | All acceptance criteria met |
| **FAIL** | ❌ | One or more criteria failed |
| **WARN** | ⚠️ | Passing but close to threshold |

---

## Gate 1: Linting

**Purpose**: Ensure code adheres to Python best practices using Ruff.

### Acceptance Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| Ruff Errors | 0 | CRITICAL |
| Ruff Warnings | 0 | HIGH |
| Complexity (McCabe) | ≤ 10 per function | HIGH |

### Validation Commands

```bash
# Run Ruff linter
poetry run ruff check .

# Run with auto-fix
poetry run ruff check --fix .

# Integration with Make
make lint
```

---

## Gate 2: Code Formatting

**Purpose**: Enforce consistent code style using Black and Ruff formatter.

### Acceptance Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| Black Compliance | 100% | CRITICAL |
| Line Length | ≤ 100 characters | CRITICAL |

### Validation Commands

```bash
# Check formatting
poetry run black --check .

# Format code
poetry run black .

# Integration with Make
make format
```

---

## Gate 3: Type Checking

**Purpose**: Enforce type safety using mypy in strict mode.

### Acceptance Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| Mypy Errors (Critical Modules) | 0 | CRITICAL |
| Type Coverage | ≥ 95% | HIGH |

**Critical Modules:**
- `src/psy_agents_noaug/data/groundtruth.py`
- `src/psy_agents_noaug/data/loaders.py`
- `src/psy_agents_noaug/training/train_loop.py`

### Validation Commands

```bash
# Run mypy
poetry run mypy src/psy_agents_noaug/data/ src/psy_agents_noaug/training/

# Integration with Make
make typecheck
```

---

## Gate 4: Unit Tests

**Purpose**: Ensure comprehensive test coverage with passing tests.

### Acceptance Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| Overall Coverage | ≥ 90% | CRITICAL |
| Test Success Rate | 100% | CRITICAL |
| Test Execution Time | ≤ 60 seconds | HIGH |

### Validation Commands

```bash
# Run all tests with coverage
poetry run pytest --cov=src/psy_agents_noaug --cov-report=html

# Run specific test file
poetry run pytest tests/test_groundtruth.py -v

# Integration with Make
make test
make test-cov
```

---

## Gate 5: Integration Tests

**Purpose**: Validate end-to-end workflows, especially augmentation pipeline.

### Acceptance Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| Integration Test Success Rate | 100% | CRITICAL |
| E2E Augmentation Pipeline | PASS | CRITICAL |
| Test Execution Time | ≤ 300 seconds | MEDIUM |

### Validation Commands

```bash
# Run integration tests
poetry run pytest -m integration -v

# Run E2E augmentation test
poetry run pytest tests/test_augmentation_e2e.py -v
```

---

## Gate 6: DataLoader Performance

**Purpose**: Ensure efficient data loading (data_time/step_time ≤ 0.40).

### Acceptance Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| Data Time Ratio | ≤ 0.40 | CRITICAL |
| Step Time (per batch) | ≤ 200ms (GPU) | HIGH |
| GPU Utilization | ≥ 90% | HIGH |

### Validation Commands

```bash
# Run benchmark script
python scripts/bench_dataloader.py --task criteria --batch-size 32

# Integration with Make
make benchmark-dataloader
```

---

## Gate 7: Dependency Security

**Purpose**: Ensure no critical/high vulnerabilities in dependencies.

### Acceptance Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| Critical Vulnerabilities | 0 | CRITICAL |
| High Vulnerabilities | 0 | CRITICAL |
| Medium Vulnerabilities | ≤ 3 | MEDIUM |

### Validation Commands

```bash
# Run Safety check
poetry run safety check --json > safety_report.json

# Or use pip-audit
poetry run pip-audit

# Integration with Make
make security-check
```

---

## Gate 8: Training Reproducibility

**Purpose**: Ensure deterministic training with identical results.

### Acceptance Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| F1 Variance (3 runs) | ≤ 0.001 | CRITICAL |
| Loss Variance | ≤ 0.01 | HIGH |
| Weight Checksum Match | 100% | CRITICAL |

### Validation Commands

```bash
# Run reproducibility test
python scripts/verify_determinism.py --seed 42 --num-runs 3

# Integration with Make
make test-reproducibility
```

---

## Gate 9: Build Artifacts

**Purpose**: Generate production-ready wheel artifacts.

### Acceptance Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| Wheel Build Success | ✅ | CRITICAL |

### Validation Commands

```bash
# Build Python wheel
poetry build
```

---

## Gate 10: Documentation

**Purpose**: Ensure comprehensive documentation.

### Acceptance Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| README Completeness | 100% | CRITICAL |
| CHANGELOG Updated | ✅ | CRITICAL |
| API Documentation | ≥ 90% coverage | HIGH |
| Broken Links | 0 | MEDIUM |

### Validation Commands

```bash
# Validate README
python scripts/validate_readme.py

# Build documentation
poetry run sphinx-build -b html docs/ docs/_build/

# Check for broken links
poetry run linkchecker docs/_build/index.html
```

---

## Common Issues & Solutions

### Issue 1: Coverage Below 90%

**Solution:**
```bash
# Generate HTML report
poetry run pytest --cov --cov-report=html

# Open htmlcov/index.html to see uncovered lines
# Add tests for critical paths first
```

### Issue 2: High Data Time Ratio (> 0.40)

**Solution:**
```python
# Increase num_workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=8)

# Enable pin_memory
dataloader = DataLoader(dataset, pin_memory=True)

# Use persistent_workers
dataloader = DataLoader(dataset, persistent_workers=True)
```

### Issue 3: Non-Deterministic Training

**Solution:**
```python
# Enable full determinism
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Production Readiness Checklist

Before production deployment:

- [ ] Gate 1 (Linting): ✅ PASS
- [ ] Gate 2 (Formatting): ✅ PASS
- [ ] Gate 3 (Type Checking): ✅ PASS
- [ ] Gate 4 (Unit Tests): ✅ PASS
- [ ] Gate 5 (Integration Tests): ✅ PASS
- [ ] Gate 6 (DataLoader Performance): ✅ PASS
- [ ] Gate 7 (Dependency Security): ✅ PASS
- [ ] Gate 8 (Training Reproducibility): ✅ PASS
- [ ] Gate 9 (Build Artifacts): ✅ PASS
- [ ] Gate 10 (Documentation): ✅ PASS

**Overall Status:** [ ] READY FOR PRODUCTION

---

**Document Owner:** [Lead Developer Name]
**Last Review:** 2025-01-25
**Next Review:** 2025-04-25 (Quarterly)
