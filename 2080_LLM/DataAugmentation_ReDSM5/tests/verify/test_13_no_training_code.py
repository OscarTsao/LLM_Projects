"""Ensure no training code imported."""
import ast
from pathlib import Path

def test_no_training_imports():
    """Augmentation code doesn't import training modules."""
    aug_files = [
        Path("src/augment/methods.py"),
        Path("src/augment/evidence.py"),
        Path("src/augment/combinator.py"),
        Path("tools/generate_augsets.py"),
    ]

    forbidden = ["src.training", "src.hpo.trainer", "pytorch_lightning"]

    for file in aug_files:
        if not file.exists():
            continue
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden_mod in forbidden:
                        assert not alias.name.startswith(forbidden_mod), \
                            f"{file}: Imports forbidden module {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for forbidden_mod in forbidden:
                    assert not module.startswith(forbidden_mod), \
                        f"{file}: Imports from forbidden module {module}"
