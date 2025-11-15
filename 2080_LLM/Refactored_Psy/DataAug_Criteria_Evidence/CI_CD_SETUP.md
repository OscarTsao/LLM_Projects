# CI/CD Testing Infrastructure - Setup Complete

## Overview

Comprehensive testing infrastructure and CI/CD pipelines have been set up for the PSY Agents AUG project with special emphasis on augmentation validation. This document provides an overview of all components and how to use them.

## What's Been Added

### 1. Test Suite Enhancement

#### Test Files Created:
- **`tests/conftest.py`**: Shared pytest fixtures and configuration
  - Sample data fixtures (posts, annotations, criterion IDs)
  - Augmentation configuration fixtures
  - Mock objects (MLflow, tokenizer, model)
  - Environment cleanup utilities
  - Random seed management

- **`tests/test_smoke.py`**: Quick sanity checks
  - Module import tests
  - Augmentation library availability checks
  - Basic operations
  - Configuration validation
  - Simple augmentation tests

- **`tests/test_integration.py`**: End-to-end workflow tests
  - Data pipeline with augmentation
  - Augmentation integration with training
  - Training pipeline
  - HPO pipeline
  - Augmentation no-leak tests
  - Reproducibility tests

#### Augmentation-Specific Tests (Existing):
- **`tests/test_augment_contract.py`**: Verify augmentation contracts
  - Deterministic behavior with seeds
  - Train-only constraint enforcement
  - Augmentation count validation
  - Disabled augmentation handling

- **`tests/test_augment_no_leak.py`**: Ensure no data leakage
  - Split isolation verification
  - Label preservation checks
  - Post-ID grouping validation

- **`tests/test_augment_pipelines.py`**: Pipeline integration
  - NLPAug pipeline tests
  - TextAttack pipeline tests
  - Multi-method augmentation

#### Other Existing Tests:
- `tests/test_groundtruth.py`: Ground truth generation
- `tests/test_loaders.py`: Data loader tests

### 2. Coverage Configuration

Added to `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["src/psy_agents_aug"]
omit = ["*/tests/*", "*/test_*.py"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    ...
]
precision = 2
show_missing = true
```

### 3. GitHub Actions CI/CD

Three workflow files in `.github/workflows/`:

#### a) **ci.yml** - Main CI Pipeline
- Runs on: Push to main/develop, pull requests
- Python versions: 3.10, 3.11
- Steps:
  1. Setup Python and Poetry
  2. Cache dependencies
  3. Install project (including augmentation libraries)
  4. Run linting (ruff, black)
  5. Run tests with coverage
  6. **Separate job for augmentation tests**
  7. Upload coverage reports
  8. Smoke tests and integration tests

#### b) **quality.yml** - Code Quality Checks
- Runs on: Push and pull requests
- Checks:
  - ruff linting
  - black formatting
  - isort import sorting
  - bandit security scanning

#### c) **release.yml** - Release Workflow
- Triggers on: Release creation
- Steps:
  1. Build distribution package
  2. Publish to TestPyPI (optional)
  3. Upload to GitHub Release

### 4. Pre-commit Hooks

Updated `.pre-commit-config.yaml` with:
- trailing-whitespace
- end-of-file-fixer
- check-yaml, check-json, check-toml
- check-merge-conflict
- detect-private-key
- ruff (with auto-fix)
- black formatting
- isort import sorting

### 5. Validation Scripts

#### `scripts/validate_installation.py`
Validates that all dependencies are installed, including augmentation libraries:
```bash
python scripts/validate_installation.py
```

Checks:
- Core dependencies (torch, transformers, pandas, etc.)
- Augmentation libraries (nlpaug, textattack)
- Project modules
- Development tools

### 6. Docker Support

#### `Dockerfile`
Multi-stage Docker build with augmentation libraries:
- Stage 1: Install dependencies with Poetry (including nlpaug, textattack)
- Stage 2: Slim runtime image
- Exposes port 5000 for MLflow

#### `docker-compose.yml`
Services:
- `app`: Main application
- `mlflow`: MLflow UI server (port 5001 to avoid conflict with NoAug)

Usage:
```bash
docker-compose up
docker-compose run app python -m psy_agents_aug.cli train
```

### 7. Documentation

- **`TESTING.md`**: Comprehensive testing guide
  - Augmentation testing guidelines
  - No-leak validation
  - Contract verification
  - How to run tests
  - Test structure
  - CI/CD information
  - Troubleshooting

## Running Tests

### Quick Commands

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test categories
pytest tests/test_smoke.py -v                    # Smoke tests
pytest tests/test_integration.py -v             # Integration tests
pytest tests/test_augment_contract.py -v        # Augmentation contracts
pytest tests/test_augment_no_leak.py -v         # No-leak validation
pytest tests/test_augment_pipelines.py -v       # Augmentation pipelines

# Run with PYTHONPATH (if not installed)
PYTHONPATH=src:$PYTHONPATH pytest tests/ -v

# Run linting
make lint

# Format code
make format
```

### Augmentation-Specific Testing

```bash
# Test augmentation contracts only
pytest tests/ -k "augment" -v

# Test with augmentation enabled
pytest tests/test_integration.py::TestAugmentationIntegration -v

# Verify no data leakage
pytest tests/test_augment_no_leak.py -v
```

### Test Coverage

Generate coverage report:
```bash
make test-cov
# View HTML report at: htmlcov/index.html
```

### Pre-commit Hooks

Install and run:
```bash
make pre-commit-install
make pre-commit-run
```

## CI/CD Badges

Add these badges to your README.md:

```markdown
![CI](https://github.com/your-org/repo/workflows/CI/badge.svg)
![Code Quality](https://github.com/your-org/repo/workflows/Code%20Quality/badge.svg)
[![codecov](https://codecov.io/gh/your-org/repo/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/repo)
```

## Configuration Files Summary

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Main CI with augmentation tests
│       ├── quality.yml               # Code quality checks
│       └── release.yml               # Release workflow
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_smoke.py                # Smoke tests
│   ├── test_integration.py          # Integration tests with augmentation
│   ├── test_augment_contract.py     # Augmentation contract tests
│   ├── test_augment_no_leak.py      # No-leak validation
│   ├── test_augment_pipelines.py    # Augmentation pipeline tests
│   ├── test_groundtruth.py          # Ground truth tests
│   └── test_loaders.py              # Data loader tests
├── scripts/
│   └── validate_installation.py
├── pyproject.toml                   # Enhanced with coverage config
├── .pre-commit-config.yaml          # Pre-commit hooks
├── Dockerfile                       # Docker build with augmentation libs
├── docker-compose.yml               # Docker services (MLflow on port 5001)
└── TESTING.md                       # Testing documentation
```

## Augmentation Testing Guidelines

### 1. Contract Verification

Always verify these contracts when testing augmentation:

- ✅ **Deterministic**: Same seed produces same results
- ✅ **Train-Only**: Only training split is augmented
- ✅ **Label Preservation**: Labels remain correct after augmentation
- ✅ **No Leakage**: No overlap between train/val/test after augmentation
- ✅ **Count Control**: Respects max_aug_per_sample limits

### 2. No-Leak Validation

Critical tests to prevent data leakage:

```python
def test_augmentation_only_on_train():
    """Verify augmentation only applies to training set."""
    # Train should augment
    train_aug, _ = pipeline.augment_batch(texts, split="train")
    assert len(train_aug) >= len(texts)
    
    # Val should NOT augment
    val_aug, _ = pipeline.augment_batch(texts, split="val")
    assert len(val_aug) == len(texts)
```

### 3. Integration Testing

Test augmentation in full pipeline:

```python
def test_augmentation_in_training_pipeline():
    """Test augmentation integrates with training."""
    # Create augmented dataloaders
    # Train model
    # Verify augmentation was applied
    # Check no leakage
```

## Testing Best Practices

1. **Always test augmentation isolation**
   ```bash
   pytest tests/test_augment_no_leak.py -v
   ```

2. **Verify determinism**
   ```python
   set_seed(42)
   result1 = augment(text)
   set_seed(42)
   result2 = augment(text)
   assert result1 == result2
   ```

3. **Test with real augmentation libraries**
   - Don't just mock augmentation
   - Verify actual nlpaug/textattack behavior
   - Check edge cases

4. **Monitor augmentation overhead**
   ```bash
   pytest tests/test_performance.py -v  # If created
   ```

## Troubleshooting

### Augmentation libraries not found

Install with Poetry:
```bash
poetry install
# Or specifically:
poetry add nlpaug textattack
```

### Augmentation tests fail

Check augmentation config:
```python
config = AugmentationConfig(
    enabled=True,
    train_only=True,  # Must be True!
    seed=42,
)
```

### Data leakage detected

Review split logic:
```bash
pytest tests/test_augment_no_leak.py -v --tb=long
```

## Augmentation-Specific Maintenance

### Adding New Augmentation Methods

1. Implement in `src/psy_agents_aug/augment/`
2. Add tests in `tests/test_augment_pipelines.py`
3. Verify contracts:
   ```bash
   pytest tests/test_augment_contract.py -v
   ```
4. Check no leakage:
   ```bash
   pytest tests/test_augment_no_leak.py -v
   ```

### Updating Augmentation Libraries

```bash
poetry update nlpaug textattack
pytest tests/test_augment_*.py -v  # Verify all augmentation tests pass
```

## Success Metrics

- ✅ All tests passing locally
- ✅ CI pipeline green on GitHub
- ✅ Code coverage > 80%
- ✅ **Augmentation coverage > 95%** (critical)
- ✅ No linting errors
- ✅ Pre-commit hooks installed
- ✅ Docker builds successfully
- ✅ **No data leakage detected**
- ✅ **Augmentation contracts verified**

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [nlpaug documentation](https://nlpaug.readthedocs.io/)
- [TextAttack documentation](https://textattack.readthedocs.io/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [Data Augmentation Best Practices](https://arxiv.org/abs/1901.11196)

---

**Setup completed on**: 2025-10-23
**Ready for production use with augmentation validation** ✨
