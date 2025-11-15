#!/bin/bash
# Verification script for CI/CD setup

set -e

echo "=========================================="
echo "CI/CD Setup Verification"
echo "=========================================="
echo ""

verify_file() {
    local file=$1
    local desc=$2
    if [ -f "$file" ]; then
        echo "✓ $desc"
        return 0
    else
        echo "✗ $desc - MISSING"
        return 1
    fi
}

verify_dir() {
    local dir=$1
    local desc=$2
    if [ -d "$dir" ]; then
        echo "✓ $desc"
        return 0
    else
        echo "✗ $desc - MISSING"
        return 1
    fi
}

check_repo() {
    local repo=$1
    echo ""
    echo "Checking $repo..."
    echo "----------------------------------------"
    
    cd "$repo"
    
    # GitHub Actions
    verify_dir ".github/workflows" "GitHub Actions workflows directory"
    verify_file ".github/workflows/ci.yml" "CI workflow"
    verify_file ".github/workflows/quality.yml" "Quality workflow"
    verify_file ".github/workflows/release.yml" "Release workflow"
    
    # Test files
    verify_file "tests/conftest.py" "Shared pytest fixtures"
    verify_file "tests/test_smoke.py" "Smoke tests"
    verify_file "tests/test_integration.py" "Integration tests"
    
    # Configuration
    verify_file "pyproject.toml" "Project configuration"
    verify_file ".pre-commit-config.yaml" "Pre-commit hooks"
    
    # Docker
    verify_file "Dockerfile" "Dockerfile"
    verify_file "docker-compose.yml" "Docker Compose"
    
    # Documentation
    verify_file "TESTING.md" "Testing documentation"
    verify_file "CI_CD_SETUP.md" "CI/CD setup guide"
    
    # Scripts
    verify_file "scripts/validate_installation.py" "Installation validator"
    
    echo ""
}

# Check NoAug repository
check_repo "/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence"

# Check DataAug repository
check_repo "/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence"

echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""
echo "Summary of Created Files:"
echo "  - 3 GitHub Actions workflows per repo"
echo "  - 3 new test files per repo"
echo "  - 1 shared fixtures file per repo"
echo "  - 2 Docker files per repo"
echo "  - 3 documentation files per repo"
echo "  - 1 validation script per repo"
echo "  - Updated pyproject.toml and pre-commit config"
echo ""
echo "Total: ~26 new/updated files across both repositories"
echo ""
echo "Next steps:"
echo "  1. Run tests: cd <repo> && PYTHONPATH=src:\$PYTHONPATH pytest tests/ -v"
echo "  2. Install pre-commit: pre-commit install"
echo "  3. Validate installation: python scripts/validate_installation.py"
echo "  4. Push to GitHub to trigger CI/CD"
echo ""
