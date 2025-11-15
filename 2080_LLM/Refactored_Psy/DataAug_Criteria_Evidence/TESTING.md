# Testing Guide

This document describes the testing infrastructure for the PSY Agents AUG project.

## Overview

The project uses `pytest` for testing with comprehensive coverage tracking. Tests include augmentation-specific validation to ensure data augmentation works correctly and doesn't leak between splits.

## Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test end-to-end workflows with augmentation
- **Smoke Tests**: Quick sanity checks for basic functionality
- **Augmentation Tests**: Verify augmentation contracts and no-leak guarantees
- **Contract Tests**: Verify API contracts and data schemas

## Quick Start

### Run All Tests

```bash
make test
```

Or directly with pytest:

```bash
poetry run pytest tests/ -v
```

### Run Tests with Coverage

```bash
make test-cov
```

This generates both terminal and HTML coverage reports. View the HTML report at `htmlcov/index.html`.

### Run Augmentation-Specific Tests

```bash
# Augmentation contract tests
poetry run pytest tests/test_augment_contract.py -v

# Augmentation no-leak tests
poetry run pytest tests/test_augment_no_leak.py -v

# Augmentation pipeline tests
poetry run pytest tests/test_augment_pipelines.py -v
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_smoke.py                  # Smoke tests (quick sanity checks)
├── test_integration.py            # Integration tests (end-to-end)
├── test_augment_contract.py       # Augmentation contract tests
├── test_augment_no_leak.py        # Augmentation no-leak tests
├── test_augment_pipelines.py      # Augmentation pipeline tests
├── test_groundtruth.py            # Ground truth generation tests
└── test_loaders.py                # Data loader tests
```

## Augmentation Testing

### Contract Tests

Verify augmentation guarantees:

1. **Deterministic**: Same seed = same results
2. **Train-Only**: Only training data is augmented
3. **Count Control**: Expected number of augmented samples

```python
def test_deterministic_augmentation(nlpaug_pipeline):
    """Test that augmentation is deterministic with same seed."""
    text = "The patient reports feeling anxious."
    
    result1 = nlpaug_pipeline.augment_text(text, num_variants=1)
    result2 = nlpaug_pipeline.augment_text(text, num_variants=1)
    
    assert result1 == result2
```

### No-Leak Tests

Ensure augmentation doesn't leak between splits:

```python
def test_train_only_constraint(nlpaug_pipeline):
    """Test that augmentation ONLY applies to training data."""
    texts = ["Test sentence one.", "Test sentence two."]
    
    # Training data should be augmented
    train_aug, _ = nlpaug_pipeline.augment_batch(texts, split="train")
    assert len(train_aug) > len(texts)
    
    # Validation data should NOT be augmented
    val_aug, _ = nlpaug_pipeline.augment_batch(texts, split="val")
    assert len(val_aug) == len(texts)
```

## Shared Fixtures

Common fixtures are defined in `tests/conftest.py`:

- `sample_posts`: Sample post data
- `sample_annotations`: Sample annotation data
- `augmentation_config`: Augmentation configuration
- `mock_mlflow`: Mocked MLflow for testing without logging
- `mock_tokenizer`: Mocked tokenizer for testing
- `mock_model`: Mocked model for testing

## Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions:

- **CI Workflow** (`.github/workflows/ci.yml`):
  - Runs on Python 3.10 and 3.11
  - Executes all tests with coverage
  - Runs augmentation-specific tests separately
  - Uploads coverage reports to Codecov

- **Quality Workflow** (`.github/workflows/quality.yml`):
  - Runs ruff, black, and isort checks
  - Performs security scanning with bandit

## Coverage Requirements

We aim for:
- Overall coverage: > 80%
- Critical paths: > 90%
- Augmentation code: > 95% (critical for correctness)

## Test Configuration

Test configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["src/psy_agents_aug"]
omit = ["*/tests/*", "*/test_*.py"]
branch = true
```

## Augmentation Validation Checklist

When adding new augmentation methods, ensure:

- [ ] Deterministic with seed
- [ ] Only applies to training split
- [ ] Preserves labels correctly
- [ ] Doesn't modify original data
- [ ] Respects max augmentation limits
- [ ] No data leakage between splits
- [ ] Compatible with batch processing

## Troubleshooting

### nlpaug Not Available

Some tests may be skipped if nlpaug is not installed:

```bash
poetry install --with dev
```

### Augmentation Tests Fail

Verify augmentation configuration:

```python
config = AugmentationConfig(
    enabled=True,
    ratio=0.5,
    max_aug_per_sample=1,
    seed=42,
    train_only=True,  # Must be True
)
```

## Best Practices

1. **Augmentation Isolation**: Test augmentation independently
2. **Split Validation**: Always verify no leakage between splits
3. **Determinism**: Use fixed seeds for reproducibility
4. **Label Preservation**: Verify labels are correctly maintained
5. **Performance**: Monitor augmentation overhead

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [nlpaug documentation](https://nlpaug.readthedocs.io/)
- [Data Augmentation Best Practices](https://arxiv.org/abs/1901.11196)
