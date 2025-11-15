# PR Plan: PSY Agents Augmentation Transformation

**Project:** Transform PSY Agents from NO-AUG baseline to AUG-enabled production system

**Strategy:** 5 sequential, independently testable Pull Requests

**Timeline:** 3 weeks (54-78 developer-hours)

**Key Insight:** 60% of augmentation infrastructure already exists, requires activation and integration

---

## Executive Summary

This plan organizes the NO-AUG → AUG transformation into 5 PRs:

1. **PR#1: Quality Gates & CI** (8-12h) - Foundation with testing/linting/security
2. **PR#2: Augmentation Integration** (16-24h) - Wire augmentation into training pipeline
3. **PR#3: HPO Integration** (12-16h) - Two-stage HPO (baseline + augmentation)
4. **PR#4: Packaging & Security** (10-14h) - packaging, SBOM, security scanning
5. **PR#5: Documentation** (8-12h) - User guides, changelog, production readiness

**Total Effort:** 54-78 hours (6.75-9.75 developer-days)

---

## PR#1: Quality Gates & CI Infrastructure

**Branch:** `feature/ci-quality-gates`

**Dependencies:** None (foundation)

**Estimated Effort:** 8-12 hours

### Objectives

Establish robust CI/CD pipeline before augmentation work begins.

### Files to Create

1. **`.pre-commit-config.yaml`** - Pre-commit hooks (ruff, mypy, pytest)
2. **`.github/workflows/ci.yaml`** - GitHub Actions (lint, typecheck, test, security)
3. **`scripts/bench_dataloader.py`** - DataLoader performance benchmark
4. **`scripts/verify_determinism.py`** - Training reproducibility verification

### Key Changes

```yaml
# .github/workflows/ci.yaml
jobs:
  lint:
    - ruff check .
    - ruff format --check .

  typecheck:
    - mypy src/ --strict

  test:
    - pytest --cov=src --cov-fail-under=90

  security:
    - pip-audit
    - bandit -r src/
```

### Acceptance Criteria

- [ ] All CI workflows pass
- [ ] Pre-commit hooks configured
- [ ] Test coverage ≥90%
- [ ] Zero high-severity vulnerabilities
- [ ] DataLoader benchmark produces results
- [ ] Determinism test passes on CPU/GPU

### Success Metrics

- CI pipeline runtime: <15 minutes
- Test coverage: ≥90%
- Security: 0 high/critical vulnerabilities
- DataLoader throughput: ≥200 samples/sec

---

## PR#2: Augmentation Integration & Core Tests

**Branch:** `feature/augmentation-integration`

**Dependencies:** PR#1 (requires CI infrastructure)

**Estimated Effort:** 16-24 hours

### Objectives

Wire augmentation into training pipeline with comprehensive tests. Activate existing 60% augmentation infrastructure.

### Files to Modify

#### 1. `src/psy_agents_noaug/data/datasets.py`

Add augmentation support:

```python
class CriteriaDataset(Dataset):
    def __init__(
        self,
        data: list[dict],
        tokenizer: PreTrainedTokenizer,
        augmenter: Optional[Augmenter] = None,
        augment_scope: str = 'train_only',
    ):
        self.augmenter = augmenter
        self.augment_scope = augment_scope

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        text = item['text']

        # Apply augmentation if enabled
        if self.augmenter and self._should_augment():
            text = self.augmenter.augment(text)

        # Tokenize...
```

#### 2. `src/psy_agents_noaug/data/loaders.py`

Pass augmenter to datasets:

```python
def create_dataloaders(
    task: str,
    augmenter: Optional[Augmenter] = None,
    augment_scope: str = 'train_only',
    **kwargs
):
    # Only augment training split if scope='train_only'
    train_augmenter = augmenter if augment_scope in ['train_only', 'all'] else None
    val_augmenter = augmenter if augment_scope == 'all' else None
```

### Files to Create (7 Test Modules)

1. **`tests/test_augmentation_registry.py`** - Registry and technique registration
2. **`tests/test_pipeline_scope.py`** - Augmentation scope (train_only/all/none)
3. **`tests/test_tfidf_cache.py`** - TF-IDF cache performance (5-10x speedup)
4. **`tests/test_seed_repro.py`** - Deterministic augmentation with seeds
5. **`tests/test_cli_aug_flags.py`** - CLI augmentation flags and overrides
6. **`tests/test_hpo_guardrails.py`** - HPO search space guardrails
7. **`tests/test_perf_contract.py`** - Performance contract (<20% overhead)

### Key Test Example

```python
# tests/test_pipeline_scope.py
def test_train_only_scope(mock_augmenter):
    """Augmentation should only apply to training split."""

    loaders = create_dataloaders(
        task='criteria',
        augmenter=mock_augmenter,
        augment_scope='train_only',
    )

    assert loaders['train'].dataset.augmenter is not None
    assert loaders['val'].dataset.augmenter is None
    assert loaders['test'].dataset.augmenter is None
```

### Acceptance Criteria

- [ ] All 7 test modules pass with ≥90% coverage
- [ ] Augmentation can be enabled/disabled via config
- [ ] Scope (train_only/all/none) works correctly
- [ ] TF-IDF cache provides ≥5x speedup
- [ ] Augmentation is deterministic with seed
- [ ] Performance overhead ≤20%
- [ ] Baseline (no augmentation) still works

---

## PR#3: HPO Integration & Observability

**Branch:** `feature/hpo-augmentation`

**Dependencies:** PR#2 (requires augmentation integration)

**Estimated Effort:** 12-16 hours

### Objectives

Enable two-stage HPO (Stage A: baseline, Stage B: augmentation) with full MLflow tracking.

### Files to Create

#### 1. `configs/hpo/stage_a_no_aug.yaml`

```yaml
# Stage A: Baseline HPO (no augmentation)
hpo:
  study_name: "stage_a_baseline_hpo"
  n_trials: 50

  search_space:
    model_name:
      type: "categorical"
      choices: ["bert-base-uncased", "roberta-base", "deberta-v3-base"]

    learning_rate:
      type: "loguniform"
      low: 1e-6
      high: 1e-4

    # ... other hyperparameters

  augmentation:
    enabled: false  # NO augmentation in Stage A
```

#### 2. `configs/hpo/stage_b_aug_only.yaml`

```yaml
# Stage B: Augmentation HPO (uses best baseline from Stage A)
hpo:
  study_name: "stage_b_augmentation_hpo"
  n_trials: 100

  # Load best baseline config
  baseline_config: "outputs/hpo_stage_a/best_config.yaml"

  search_space:
    # Only search augmentation hyperparameters
    synonym_replacement:
      enabled: {type: "categorical", choices: [true, false]}
      prob: {type: "uniform", low: 0.05, high: 0.3}
      n: {type: "int", low: 1, high: 5}

    contextual_word_embeddings:
      enabled: {type: "categorical", choices: [true, false]}
      prob: {type: "uniform", low: 0.05, high: 0.2}

    # Constraint: at most 2 techniques enabled
  constraints:
    - "sum([synonym.enabled, contextual.enabled]) <= 2"
```

#### 3. `scripts/compare_stages.py`

Generate comparison report:

```python
def compare_stages():
    stage_a = load_best_run('stage_a_baseline_hpo')
    stage_b = load_best_run('stage_b_augmentation_hpo')

    improvement = (stage_b['test_f1'] - stage_a['test_f1']) / stage_a['test_f1']

    report = f"""
Stage A (Baseline): {stage_a['test_f1']:.4f}
Stage B (Augmented): {stage_b['test_f1']:.4f}
Improvement: {improvement:+.2%}

Conclusion: {'✅ Augmentation helps' if improvement > 0.01 else '⚠️ No benefit'}
"""
    print(report)
```

### Files to Modify

#### `scripts/tune_max.py`

Add augmentation search space:

```python
def get_search_space(trial, stage='baseline'):
    if stage == 'augmentation':
        # Load baseline config
        baseline = load_baseline_config()

        # Add augmentation params
        params = baseline.copy()
        params['augmentation'] = {
            'enabled': True,
            'synonym_prob': trial.suggest_uniform('syn_prob', 0.05, 0.3),
            # ...
        }
        return params
```

### Acceptance Criteria

- [ ] Stage A runs without augmentation
- [ ] Stage B uses best baseline from Stage A
- [ ] Augmentation params logged to MLflow
- [ ] Comparison report generated
- [ ] Both stages complete within timeout

---

## PR#4: Packaging & Security

**Branch:** `feature/production-deployment`

**Dependencies:** PR#3 (requires complete HPO system)

**Estimated Effort:** 10-14 hours

### Objectives

Create production deployment artifacts focused on distribution packaging, SBOM generation, and security scanning (container build no longer in scope).

### Files to Create

#### 1. `scripts/generate_sbom.py`

Generate Software Bill of Materials:

```python
def generate_cyclonedx_sbom():
    packages = get_installed_packages()

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "components": [
            {
                "type": "library",
                "name": pkg['name'],
                "version": pkg['version'],
                "purl": f"pkg:pypi/{pkg['name']}@{pkg['version']}"
            }
            for pkg in packages
        ]
    }

    return sbom
```

#### 2. `scripts/audit_security.py`

Security audit with failure on high/critical:

```python
def main():
    results = run_pip_audit()
    by_severity = analyze_vulnerabilities(results)

    if by_severity['critical'] or by_severity['high']:
        print("❌ FAIL: Critical/high vulnerabilities found")
        sys.exit(1)

    print("✅ PASS: No critical/high vulnerabilities")
```

### Acceptance Criteria

- [ ] SBOM generated in CycloneDX format
- [ ] Security audit: 0 high/critical vulnerabilities
- [ ] Wheel artifact builds successfully via `poetry build`

---

## PR#5: Documentation & Release

**Branch:** `feature/documentation-release`

**Dependencies:** PR#4 (requires all infrastructure complete)

**Estimated Effort:** 8-12 hours

### Objectives

Create comprehensive documentation and production readiness report.

### Files to Create

#### 1. `CHANGELOG.md`

```markdown
# Changelog

## [0.2.0] - 2025-01-XX

### Added
- Data augmentation system (3 techniques)
- Two-stage HPO (baseline + augmentation)
- CI/CD infrastructure
- Distribution packaging automation
- Security scanning and SBOM

### Changed
- Datasets now accept optional augmenter
- CLI supports augmentation flags

[0.2.0]: https://github.com/.../compare/v0.1.0...v0.2.0
```

#### 2. `CONTRIBUTING.md`

Developer guide with setup, workflow, testing, code style.

#### 3. `docs/AUGMENTATION_GUIDE.md`

Comprehensive user guide:

```markdown
# Augmentation Guide

## Quick Start

```yaml
augmentation:
  enabled: true
  scope: train_only
  techniques:
    synonym_replacement:
      prob: 0.1
      n: 2
```

## Techniques

1. **Synonym Replacement** - TF-IDF weighted
2. **Contextual Embeddings** - BERT-based
3. **Back Translation** - Experimental

## Performance

- Overhead: <20%
- Cache speedup: 5-10x
```

#### 4. `PRODUCTION_READINESS_REPORT.md`

Sign-off document with quality gate status, metrics, deployment checklist.

### Files to Modify

#### `README.md`

Add augmentation section:

```markdown
## Data Augmentation 🚀

PSY Agents now supports data augmentation:

```bash
# Train with augmentation
make train

# HPO with augmentation
make hpo-stage-a  # Baseline
make hpo-stage-b  # Augmentation
```

See [AUGMENTATION_GUIDE.md](docs/AUGMENTATION_GUIDE.md) for details.
```

### Acceptance Criteria

- [ ] CHANGELOG.md complete
- [ ] CONTRIBUTING.md comprehensive
- [ ] AUGMENTATION_GUIDE.md covers all use cases
- [ ] PRODUCTION_READINESS_REPORT.md signed off
- [ ] README.md updated
- [ ] All links working

---

## Cross-PR Dependencies

```
PR#1 (CI/CD) ──┐
               ├──> PR#2 (Augmentation) ──> PR#3 (HPO) ──> PR#4 (Packaging) ──> PR#5 (Docs)
main ──────────┘
```

**Critical Path:** PR#1 → PR#2 → PR#3 → PR#4 → PR#5

---

## Timeline

### Week 1 (Jan 6-12)
- **Mon-Tue:** PR#1 (CI/CD) - 12 hours
- **Wed-Fri:** PR#2 (Augmentation) - 24 hours
- **Total:** 36 hours

### Week 2 (Jan 13-19)
- **Mon-Wed:** PR#3 (HPO) - 16 hours
- **Thu-Fri:** PR#4 (Packaging) - 14 hours
- **Total:** 30 hours

### Week 3 (Jan 20-26)
- **Mon-Tue:** PR#5 (Docs) - 12 hours
- **Wed:** Final validation - 4 hours
- **Thu-Fri:** Buffer - 8 hours
- **Total:** 24 hours

**Grand Total:** 66 hours planned + 12 hours buffer = 78 hours (9.75 days)

---

## Success Criteria

### Technical
- [ ] All 5 PRs merged to main
- [ ] All CI checks passing
- [ ] Test coverage ≥90%
- [ ] Security audit clean
- [ ] Release artifacts build and install cleanly

### Quality
- [ ] Documentation complete
- [ ] Code review approval (1+ per PR)
- [ ] Performance benchmarks met
- [ ] No regression in baseline metrics

### Process
- [ ] Timeline met (±3 days acceptable)
- [ ] No blocking issues
- [ ] Production readiness report signed off

---

## Risk Mitigation

### PR#1 Risks
**Risk:** CI failures block development
**Mitigation:** Test locally first, use matrix testing

### PR#2 Risks
**Risk:** Augmentation breaks baseline
**Mitigation:** Keep `augmenter=None` default, extensive testing

### PR#3 Risks
**Risk:** Stage B doesn't improve over Stage A
**Mitigation:** Document "no improvement" as valid result

### PR#4 Risks
**Risk:** Security vulnerabilities block deployment
**Mitigation:** Regular `poetry update`, pin vulnerable packages

### PR#5 Risks
**Risk:** Incomplete documentation blocks adoption
**Mitigation:** Multiple reviewers, comprehensive checklist

---

**Document Version:** 1.0
**Owner:** [Lead Developer Name]
**Status:** APPROVED
