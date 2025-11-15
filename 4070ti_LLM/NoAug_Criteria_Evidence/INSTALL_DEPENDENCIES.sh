#!/bin/bash
# Install required dependencies for MLflow enhancements

echo "================================================================================"
echo "Installing MLflow Enhancement Dependencies"
echo "================================================================================"
echo ""

cd "$(dirname "$0")"

echo "Current directory: $(pwd)"
echo ""

# Check if poetry is available
if ! command -v poetry &> /dev/null; then
    echo "ERROR: Poetry not found. Please install poetry first."
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

echo "Poetry version: $(poetry --version)"
echo ""

echo "Installing dependencies..."
echo ""

# Install psutil for CPU/memory monitoring
echo "1. Installing psutil (CPU/memory monitoring)..."
poetry add psutil

echo ""

# Install pynvml for detailed GPU monitoring
echo "2. Installing pynvml (detailed GPU monitoring)..."
poetry add pynvml

echo ""

# Verify scikit-learn (should already be present)
echo "3. Verifying scikit-learn..."
poetry show scikit-learn

echo ""

# Verify pandas (should already be present)
echo "4. Verifying pandas..."
poetry show pandas

echo ""

# Verify mlflow
echo "5. Verifying mlflow..."
poetry show mlflow

echo ""
echo "================================================================================"
echo "Dependency Installation Complete"
echo "================================================================================"
echo ""

echo "Installed packages:"
echo "  [NEW] psutil     - CPU/memory monitoring"
echo "  [NEW] pynvml     - Detailed GPU metrics"
echo "  [OK]  scikit-learn - Metrics (already present)"
echo "  [OK]  pandas     - Data handling (already present)"
echo "  [OK]  mlflow     - Experiment tracking (already present)"
echo ""

echo "Next steps:"
echo "  1. Verify installation: poetry show psutil pynvml"
echo "  2. Test training: python -m psy_agents_noaug.cli train task=criteria training.num_epochs=2"
echo "  3. View MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo ""

echo "For documentation, see:"
echo "  - MLFLOW_ENHANCEMENTS_GUIDE.md (complete guide)"
echo "  - MLFLOW_QUICK_REFERENCE.md (quick start)"
echo "  - MLFLOW_ENHANCEMENT_SUMMARY.md (implementation details)"
echo ""

echo "Installation complete! ✅"
