.PHONY: install lint format typecheck test test-smoke test-coverage verify clean help

help:
	@echo "ReDSM5 Build Automation"
	@echo "======================="
	@echo ""
	@echo "Available targets:"
	@echo "  make install        - Install project with dev dependencies"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Format code with black"
	@echo "  make typecheck      - Run mypy type checker"
	@echo "  make test           - Run fast tests with coverage"
	@echo "  make test-smoke     - Run integration smoke tests"
	@echo "  make test-coverage  - Run tests and generate HTML coverage report"
	@echo "  make verify         - Run full verification suite"
	@echo "  make clean          - Remove generated files"
	@echo "  make check-env      - Verify environment setup"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	pip check

lint:
	@echo "Running ruff..."
	ruff check src/ tests/

format:
	@echo "Formatting with black..."
	black src/ tests/
	@echo "Sorting imports with isort..."
	isort src/ tests/

typecheck:
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports

test:
	@echo "Running fast tests..."
	pytest tests/ -v -m "not slow" --cov=src --cov-report=term --cov-report=html

test-smoke:
	@echo "Running integration smoke tests..."
	pytest tests/ -v -m "slow or integration" --tb=short

test-coverage:
	@echo "Running tests with HTML coverage report..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

verify: lint typecheck test
	@echo "Running verification report generation..."
	python -m scripts.build_verification_report
	@echo ""
	@echo "✅ Verification complete!"
	@echo "   Reports: VERIFICATION_REPORT.md, VERIFICATION_SUMMARY.json"

check-env:
	@echo "Checking environment..."
	python scripts/check_env.py

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.json .coverage.*
	rm -rf *.egg-info build dist
	rm -rf outputs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"
