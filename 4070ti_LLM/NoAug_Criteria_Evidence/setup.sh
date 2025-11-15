#!/bin/bash
# Setup script for PSY Agents NO-AUG repository

set -e

echo "=================================================="
echo "PSY Agents NO-AUG Repository Setup"
echo "=================================================="

# Check Python version
echo -e "\nChecking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if ! [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "ERROR: Python 3.10+ required, found Python $python_version"
    exit 1
fi
echo "✓ Python $python_version found"

# Install Poetry if not present
if ! command -v poetry &> /dev/null; then
    echo -e "\nPoetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo "✓ Poetry installed"
else
    echo -e "\n✓ Poetry already installed"
fi

# Install dependencies
echo -e "\nInstalling dependencies..."
poetry install --with dev
echo "✓ Dependencies installed"

# Install pre-commit hooks
echo -e "\nInstalling pre-commit hooks..."
poetry run pre-commit install
echo "✓ Pre-commit hooks installed"

# Verify directory structure
echo -e "\nVerifying directory structure..."
required_dirs=(
    "src/psy_agents_noaug/data"
    "src/psy_agents_noaug/models"
    "src/psy_agents_noaug/training"
    "src/psy_agents_noaug/hpo"
    "src/psy_agents_noaug/utils"
    "configs/data"
    "configs/model"
    "configs/training"
    "configs/hpo"
    "configs/task"
    "data/raw/redsm5"
    "data/processed"
    "scripts"
    "tests"
)

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "  ✗ Missing directory: $dir"
        exit 1
    fi
done
echo "✓ Directory structure verified"

# Check for DSM criteria file
echo -e "\nChecking for DSM criteria file..."
if [ ! -f "data/raw/redsm5/dsm_criteria.json" ]; then
    echo "  ✗ DSM criteria file not found: data/raw/redsm5/dsm_criteria.json"
    echo "  Please copy the DSM criteria JSON file to this location"
else
    echo "✓ DSM criteria file found"
fi

# Run tests
echo -e "\nRunning tests..."
poetry run pytest tests/ -v --tb=short
echo "✓ All tests passed"

echo -e "\n=================================================="
echo "Setup complete!"
echo "=================================================="
echo -e "\nNext steps:"
echo "  1. Add your data to data/processed/"
echo "  2. Generate ground truth: make groundtruth-criteria INPUT=<path>"
echo "  3. Run HPO: make hpo-sanity TASK=criteria MODEL=bert_base"
echo "  4. Train model: make train-best STUDY=<path> TASK=criteria"
echo ""
echo "For more information, see README.md"
