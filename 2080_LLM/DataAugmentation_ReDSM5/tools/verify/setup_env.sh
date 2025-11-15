#!/bin/bash
# Verification environment setup
set -e
echo "=== Verification Suite Setup ==="
python3 --version || (echo "Python 3 required" && exit 1)
pip install -q pytest>=7.4 pytest-json-report>=1.5 pytest-xdist>=3.3 ruff>=0.1 mypy>=1.0
echo "✓ Dev dependencies installed"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "CUDA check skipped (torch not installed)"
echo "✓ Setup complete"
