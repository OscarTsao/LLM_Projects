# CI/CD Testing Infrastructure - Setup Complete

## Overview

Comprehensive testing infrastructure and CI/CD pipelines have been set up for the PSY Agents NO-AUG project. This document provides an overview of all components and how to use them.

## What's Been Added

### 1. Test Suite Enhancement

#### Test Files Created:
- **`tests/conftest.py`**: Shared pytest fixtures and configuration
  - Sample data fixtures (posts, annotations, criterion IDs)
  - Mock objects (MLflow, tokenizer, model)
  - Environment cleanup utilities
  - Random seed management

- **`tests/test_smoke.py`**: Quick sanity checks
  - Module import tests
  - Basic operations (tensor, dataframe creation)
  - Configuration validation
  - Environment checks (PyTorch, CUDA, transformers)

- **`tests/test_integration.py`**: End-to-end workflow tests
  - Data pipeline (groundtruth → dataloaders)
  - Training pipeline
  - HPO pipeline
  - CLI integration
  - Reproducibility tests

#### Existing Tests Enhanced:
- `tests/test_groundtruth.py`: Ground truth generation with strict validation
- `tests/test_loaders.py`: Data loader tests
- `tests/test_training_smoke.py`: Training smoke tests
- `tests/test_hpo_config.py`: HPO configuration tests

### 2. Coverage Configuration

Added to `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["src/psy_agents_noaug"]
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
  3. Install project
  4. Run linting (ruff, black)
  5. Run tests with coverage
  6. Upload coverage reports
  7. Separate jobs for smoke tests and integration tests

#### b) **quality.yml** - Code Quality Checks
- Runs on: Push and pull requests
- Checks:
  - ruff linting
  - black formatting
  - isort import sorting
  - bandit security scanning
  - TODO comment tracking

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
Validates that all dependencies are installed:
```bash
python scripts/validate_installation.py
```

Checks:
- Core dependencies (torch, transformers, pandas, etc.)
- Project modules
- Development tools

### 6. Docker Support

#### `Dockerfile`
Multi-stage Docker build:
- Stage 1: Install dependencies with Poetry
- Stage 2: Slim runtime image
- Exposes port 5000 for MLflow

#### `docker-compose.yml`
Services:
- `app`: Main application
- `mlflow`: MLflow UI server

Usage:
```bash
docker-compose up
docker-compose run app python -m psy_agents_noaug.cli train
```

### 7. Documentation

- **`TESTING.md`**: Comprehensive testing guide
  - How to run tests
  - Test structure
  - Writing tests
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
pytest tests/test_smoke.py -v              # Smoke tests
pytest tests/test_integration.py -v       # Integration tests
pytest tests/test_groundtruth.py -v       # Ground truth tests

# Run with PYTHONPATH (if not installed)
PYTHONPATH=src:$PYTHONPATH pytest tests/ -v

# Run linting
make lint

# Format code
make format
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
│       ├── ci.yml              # Main CI pipeline
│       ├── quality.yml         # Code quality checks
│       └── release.yml         # Release workflow
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── test_smoke.py          # Smoke tests
│   ├── test_integration.py    # Integration tests
│   ├── test_groundtruth.py    # Ground truth tests
│   ├── test_loaders.py        # Data loader tests
│   ├── test_training_smoke.py # Training tests
│   └── test_hpo_config.py     # HPO tests
├── scripts/
│   └── validate_installation.py
├── pyproject.toml             # Enhanced with coverage config
├── .pre-commit-config.yaml    # Pre-commit hooks
├── Dockerfile                 # Docker build
├── docker-compose.yml         # Docker services
└── TESTING.md                 # Testing documentation
```

## Testing Best Practices

1. **Run tests locally before pushing**
   ```bash
   make test
   make lint
   ```

2. **Keep tests fast**
   - Unit tests should run in < 1 second
   - Use mocks for external dependencies
   - Mark slow tests with `@pytest.mark.slow`

3. **Write descriptive test names**
   ```python
   def test_criteria_groundtruth_enforces_status_field():
       """Test that criteria groundtruth enforces using status field."""
   ```

4. **Use fixtures for common setup**
   ```python
   def test_something(sample_posts, mock_mlflow):
       # sample_posts and mock_mlflow are auto-provided
   ```

5. **Test in isolation**
   - No dependencies between tests
   - Clean up after tests
   - Use temporary directories

## Troubleshooting

### Tests fail with "ModuleNotFoundError"

Set PYTHONPATH:
```bash
export PYTHONPATH=/path/to/project/src:$PYTHONPATH
```

Or install in development mode:
```bash
pip install -e .
```

### Coverage report not generating

Ensure pytest-cov is installed:
```bash
pip install pytest-cov
```

### Pre-commit hooks failing

Update hooks:
```bash
pre-commit autoupdate
pre-commit run --all-files
```

### Docker build fails

Check Poetry lock file is up to date:
```bash
poetry lock --no-update
```

## Next Steps

1. **Push to GitHub** to trigger CI/CD pipelines
2. **Review test results** in GitHub Actions
3. **Set up Codecov** for coverage tracking (optional)
4. **Configure branch protection** requiring CI to pass
5. **Add CI/CD badges** to README
6. **Monitor test performance** and optimize slow tests

## Maintenance

### Adding New Tests

1. Create test file in `tests/` directory
2. Import fixtures from `conftest.py`
3. Follow naming convention: `test_*.py`
4. Run locally: `pytest tests/test_new.py -v`
5. Update `TESTING.md` if needed

### Updating Dependencies

```bash
poetry add <package>
poetry lock
poetry install
pytest tests/ -v  # Verify tests still pass
```

### Updating CI Workflows

1. Edit `.github/workflows/*.yml`
2. Test locally with [act](https://github.com/nektos/act) (optional)
3. Push and monitor GitHub Actions
4. Review logs and adjust as needed

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [Poetry documentation](https://python-poetry.org/docs/)
- [Pre-commit documentation](https://pre-commit.com/)

## Success Metrics

- ✅ All tests passing locally
- ✅ CI pipeline green on GitHub
- ✅ Code coverage > 80%
- ✅ No linting errors
- ✅ Pre-commit hooks installed
- ✅ Docker builds successfully

---

**Setup completed on**: 2025-10-23
**Ready for production use** ✨
