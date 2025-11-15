# Testing Guide

This document describes the testing infrastructure for the PSY Agents NO-AUG project.

## Overview

The project uses `pytest` for testing with comprehensive coverage tracking. Tests are organized into several categories:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test end-to-end workflows
- **Smoke Tests**: Quick sanity checks for basic functionality
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

### Run Specific Test Categories

```bash
# Smoke tests only
poetry run pytest tests/test_smoke.py -v

# Integration tests only
poetry run pytest tests/test_integration.py -v

# Ground truth tests only
poetry run pytest tests/test_groundtruth.py -v

# Data loader tests only
poetry run pytest tests/test_loaders.py -v
```

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── test_smoke.py              # Smoke tests (quick sanity checks)
├── test_integration.py        # Integration tests (end-to-end)
├── test_groundtruth.py        # Ground truth generation tests
├── test_loaders.py            # Data loader tests
├── test_training_smoke.py     # Training smoke tests
├── test_hpo_config.py         # Legacy HPO configuration tests
├── test_hpo_max_smoke.py      # Maximal HPO smoke regression
└── test_hpo_stage_smoke.py    # Multi-stage HPO smoke regression
```

## Shared Fixtures

Common fixtures are defined in `tests/conftest.py`:

- `sample_posts`: Sample post data
- `sample_annotations`: Sample annotation data
- `valid_criterion_ids`: Valid criterion IDs
- `field_map_path`: Temporary field map configuration
- `mock_mlflow`: Mocked MLflow for testing without logging
- `mock_tokenizer`: Mocked tokenizer for testing
- `mock_model`: Mocked model for testing

### HPO Smoke Mode

The new HPO smoke tests set `HPO_SMOKE_MODE=1`, which short‑circuits the training
pipeline and returns deterministic metrics. This keeps `pytest` fast and avoids
external downloads while still exercising the orchestration logic.

## Writing Tests

### Unit Test Example

```python
def test_normalize_status_value(field_map_path):
    """Test status value normalization."""
    from psy_agents_noaug.data.groundtruth import load_field_map, normalize_status_value

    field_map = load_field_map(field_map_path)
    status_map = field_map['status_values']

    assert normalize_status_value('positive', status_map) == 1
    assert normalize_status_value('negative', status_map) == 0
```

### Integration Test Example

```python
def test_data_pipeline_end_to_end(sample_posts, sample_annotations):
    """Test complete data pipeline."""
    # Generate ground truth
    criteria_gt = create_criteria_groundtruth(...)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_criteria_dataloaders(...)

    # Verify loaders work
    assert train_loader is not None
```

## Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions:

- **CI Workflow** (`.github/workflows/ci.yml`):
  - Runs on Python 3.10 and 3.11
  - Executes all tests with coverage
  - Uploads coverage reports to Codecov
  - Runs linting checks

- **Quality Workflow** (`.github/workflows/quality.yml`):
  - Runs ruff, black, and isort checks
  - Performs security scanning with bandit

## Coverage Requirements

We aim for:
- Overall coverage: > 80%
- Critical paths: > 90%

Current coverage can be viewed in the HTML report or by running:

```bash
poetry run coverage report
```

## Test Configuration

Test configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["src/psy_agents_noaug"]
omit = ["*/tests/*", "*/test_*.py"]
branch = true
```

## Troubleshooting

### Tests Fail Due to Missing Dependencies

```bash
poetry install --with dev
```

### Import Errors

Make sure the package is installed in editable mode:

```bash
poetry install
```

### MLflow/Optuna Database Locks

Clean up test artifacts:

```bash
make clean
```

### CUDA/GPU Tests

GPU tests are automatically skipped if CUDA is not available. To run GPU-specific tests:

```bash
poetry run pytest tests/ -v -m gpu
```

## Best Practices

1. **Isolation**: Tests should not depend on each other
2. **Reproducibility**: Use fixed random seeds in tests
3. **Fast**: Keep unit tests fast (< 1 second each)
4. **Clear Names**: Test names should describe what they test
5. **Fixtures**: Use fixtures for common setup
6. **Mocking**: Mock external dependencies (MLflow, APIs, etc.)
7. **Coverage**: Aim for high coverage but don't sacrifice test quality

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
