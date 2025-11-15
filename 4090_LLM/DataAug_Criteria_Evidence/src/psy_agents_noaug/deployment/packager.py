#!/usr/bin/env python
"""Model packaging utilities (Phase 14).

This module provides utilities for creating deployment packages
from trained models.

Key Features:
- Create self-contained deployment packages
- Include dependencies and requirements
- Configuration management
- Docker support
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class PackageConfig:
    """Configuration for deployment package."""

    model_name: str
    version: str
    created_at: str
    python_version: str
    torch_version: str
    cuda_version: str | None
    dependencies: list[str]
    metadata: dict[str, Any]


class DeploymentPackager:
    """Create deployment packages for models."""

    def __init__(self):
        """Initialize deployment packager."""
        LOGGER.info("Initialized DeploymentPackager")

    def create_deployment_package(
        self,
        model_path: Path | str,
        output_dir: Path | str,
        model_name: str,
        version: str,
        config: dict[str, Any] | None = None,
        include_dependencies: bool = True,
    ) -> Path:
        """Create deployment package.

        Args:
            model_path: Path to model checkpoint
            output_dir: Output directory for package
            model_name: Model name
            version: Model version
            config: Model configuration
            include_dependencies: Include requirements.txt

        Returns:
            Path to created package
        """
        model_path = Path(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        package_dir = output_dir / f"{model_name}-v{version}"
        package_dir.mkdir(exist_ok=True)

        LOGGER.info("Creating deployment package: %s", package_dir)

        # Copy model
        model_dest = package_dir / "model.pt"
        shutil.copy2(model_path, model_dest)
        LOGGER.info("Copied model to: %s", model_dest)

        # Save configuration
        if config:
            config_path = package_dir / "config.json"
            with config_path.open("w") as f:
                json.dump(config, f, indent=2)
            LOGGER.info("Saved configuration to: %s", config_path)

        # Create package config
        package_config = self._create_package_config(
            model_name=model_name,
            version=version,
            config=config or {},
        )

        config_path = package_dir / "package.json"
        with config_path.open("w") as f:
            json.dump(asdict(package_config), f, indent=2)

        # Create requirements.txt
        if include_dependencies:
            self._create_requirements(package_dir)

        # Create inference script
        self._create_inference_script(package_dir)

        # Create README
        self._create_readme(package_dir, model_name, version)

        LOGGER.info("Created deployment package: %s", package_dir)
        return package_dir

    def _create_package_config(
        self,
        model_name: str,
        version: str,
        config: dict[str, Any],
    ) -> PackageConfig:
        """Create package configuration.

        Args:
            model_name: Model name
            version: Version
            config: Model config

        Returns:
            Package configuration
        """
        import sys
        from datetime import datetime

        return PackageConfig(
            model_name=model_name,
            version=version,
            created_at=datetime.now().isoformat(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
            dependencies=self._get_dependencies(),
            metadata=config,
        )

    @staticmethod
    def _get_dependencies() -> list[str]:
        """Get package dependencies.

        Returns:
            List of dependencies
        """
        return [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "numpy>=1.24.0",
            "mlflow>=2.0.0",
        ]

    def _create_requirements(self, package_dir: Path) -> None:
        """Create requirements.txt.

        Args:
            package_dir: Package directory
        """
        requirements_path = package_dir / "requirements.txt"
        requirements = self._get_dependencies()

        with requirements_path.open("w") as f:
            for req in requirements:
                f.write(f"{req}\n")

        LOGGER.info("Created requirements.txt")

    def _create_inference_script(self, package_dir: Path) -> None:
        """Create inference script.

        Args:
            package_dir: Package directory
        """
        script_path = package_dir / "inference.py"

        script_content = '''#!/usr/bin/env python
"""Model inference script."""

import torch
from pathlib import Path


class ModelInference:
    """Simple inference wrapper."""

    def __init__(self, model_path: str = "model.pt", device: str = "cpu"):
        """Initialize inference.

        Args:
            model_path: Path to model
            device: Device to use
        """
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()

    def predict(self, inputs):
        """Run prediction.

        Args:
            inputs: Model inputs

        Returns:
            Predictions
        """
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs


if __name__ == "__main__":
    # Example usage
    inference = ModelInference()
    print("Model loaded successfully")
'''

        with script_path.open("w") as f:
            f.write(script_content)

        LOGGER.info("Created inference script")

    def _create_readme(
        self,
        package_dir: Path,
        model_name: str,
        version: str,
    ) -> None:
        """Create README.

        Args:
            package_dir: Package directory
            model_name: Model name
            version: Version
        """
        readme_path = package_dir / "README.md"

        readme_content = f"""# {model_name} v{version}

Deployment package for {model_name} model.

## Contents

- `model.pt`: Model checkpoint
- `config.json`: Model configuration
- `package.json`: Package metadata
- `requirements.txt`: Python dependencies
- `inference.py`: Inference script

## Usage

```python
from inference import ModelInference

# Load model
inference = ModelInference(model_path="model.pt", device="cuda")

# Run inference
predictions = inference.predict(inputs)
```

## Installation

```bash
pip install -r requirements.txt
```

## Model Details

- Name: {model_name}
- Version: {version}
- Created: See package.json
"""

        with readme_path.open("w") as f:
            f.write(readme_content)

        LOGGER.info("Created README")


def create_deployment_package(
    model_path: Path | str,
    output_dir: Path | str,
    model_name: str,
    version: str,
    **kwargs: Any,
) -> Path:
    """Create deployment package (convenience function).

    Args:
        model_path: Path to model
        output_dir: Output directory
        model_name: Model name
        version: Version
        **kwargs: Additional arguments

    Returns:
        Path to package
    """
    packager = DeploymentPackager()
    return packager.create_deployment_package(
        model_path=model_path,
        output_dir=output_dir,
        model_name=model_name,
        version=version,
        **kwargs,
    )
