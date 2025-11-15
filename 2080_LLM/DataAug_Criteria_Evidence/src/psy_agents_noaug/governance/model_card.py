#!/usr/bin/env python
"""Model cards for documentation (Phase 25).

This module provides tools for creating standardized model documentation
following the Model Cards framework.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelCard:
    """Standardized model documentation card."""

    # Model details
    model_name: str
    model_version: str
    model_type: str
    model_description: str
    created_at: datetime = field(default_factory=datetime.now)

    # Model developers
    developers: list[str] = field(default_factory=list)
    contact: str = ""

    # Intended use
    intended_use: str = ""
    intended_users: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)

    # Training data
    training_data_description: str = ""
    training_data_size: int = 0
    training_data_source: str = ""

    # Model architecture
    architecture: str = ""
    input_format: str = ""
    output_format: str = ""

    # Performance metrics
    metrics: dict[str, float] = field(default_factory=dict)
    evaluation_data_description: str = ""

    # Ethical considerations
    ethical_considerations: list[str] = field(default_factory=list)
    fairness_assessment: str = ""
    bias_risks: list[str] = field(default_factory=list)

    # Limitations
    limitations: list[str] = field(default_factory=list)
    known_issues: list[str] = field(default_factory=list)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert model card to dictionary.

        Returns:
            Dictionary representation
        """
        card_dict = asdict(self)
        # Convert datetime to ISO format
        card_dict["created_at"] = self.created_at.isoformat()
        return card_dict

    def to_json(self, indent: int = 2) -> str:
        """Convert model card to JSON string.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: Path | str) -> None:
        """Save model card to file.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(self.to_json())

        LOGGER.info(f"Saved model card to {filepath}")

    @classmethod
    def load(cls, filepath: Path | str) -> ModelCard:
        """Load model card from file.

        Args:
            filepath: Path to model card file

        Returns:
            Loaded model card
        """
        filepath = Path(filepath)

        with open(filepath) as f:
            data = json.load(f)

        # Convert ISO format back to datetime
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return cls(**data)


class ModelCardGenerator:
    """Generator for creating model cards."""

    def __init__(self):
        """Initialize model card generator."""
        LOGGER.info("Initialized ModelCardGenerator")

    def create_card(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        description: str,
        **kwargs: Any,
    ) -> ModelCard:
        """Create a new model card.

        Args:
            model_name: Model name
            model_version: Model version
            model_type: Type of model
            description: Model description
            **kwargs: Additional card properties

        Returns:
            Created model card
        """
        card = ModelCard(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            model_description=description,
            **kwargs,
        )

        LOGGER.info(f"Created model card for {model_name} v{model_version}")
        return card

    def add_training_info(
        self,
        card: ModelCard,
        data_description: str,
        data_size: int,
        data_source: str = "",
    ) -> None:
        """Add training data information to card.

        Args:
            card: Model card
            data_description: Description of training data
            data_size: Number of training samples
            data_source: Source of training data
        """
        card.training_data_description = data_description
        card.training_data_size = data_size
        card.training_data_source = data_source

        LOGGER.info(f"Added training info: {data_size} samples")

    def add_performance_metrics(
        self,
        card: ModelCard,
        metrics: dict[str, float],
        evaluation_description: str = "",
    ) -> None:
        """Add performance metrics to card.

        Args:
            card: Model card
            metrics: Performance metrics
            evaluation_description: Description of evaluation data
        """
        card.metrics = metrics
        card.evaluation_data_description = evaluation_description

        LOGGER.info(f"Added {len(metrics)} performance metrics")

    def add_ethical_considerations(
        self,
        card: ModelCard,
        considerations: list[str],
        fairness_assessment: str = "",
        bias_risks: list[str] | None = None,
    ) -> None:
        """Add ethical considerations to card.

        Args:
            card: Model card
            considerations: List of ethical considerations
            fairness_assessment: Fairness assessment description
            bias_risks: List of potential bias risks
        """
        card.ethical_considerations = considerations
        card.fairness_assessment = fairness_assessment
        card.bias_risks = bias_risks or []

        LOGGER.info(f"Added {len(considerations)} ethical considerations")

    def add_limitations(
        self,
        card: ModelCard,
        limitations: list[str],
        known_issues: list[str] | None = None,
    ) -> None:
        """Add model limitations to card.

        Args:
            card: Model card
            limitations: List of limitations
            known_issues: List of known issues
        """
        card.limitations = limitations
        card.known_issues = known_issues or []

        LOGGER.info(f"Added {len(limitations)} limitations")

    def generate_html_report(self, card: ModelCard) -> str:
        """Generate HTML report from model card.

        Args:
            card: Model card

        Returns:
            HTML report string
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Card: {card.model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        .section {{ margin-bottom: 30px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
        ul {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <h1>{card.model_name} v{card.model_version}</h1>
    <p><strong>Type:</strong> {card.model_type}</p>
    <p><strong>Created:</strong> {card.created_at.strftime('%Y-%m-%d')}</p>

    <div class="section">
        <h2>Description</h2>
        <p>{card.model_description}</p>
    </div>

    <div class="section">
        <h2>Intended Use</h2>
        <p>{card.intended_use or 'Not specified'}</p>
        {"<p><strong>Intended users:</strong> " + ", ".join(card.intended_users) + "</p>" if card.intended_users else ""}
    </div>

    <div class="section">
        <h2>Training Data</h2>
        <p>{card.training_data_description or 'Not specified'}</p>
        {"<p><strong>Size:</strong> " + str(card.training_data_size) + " samples</p>" if card.training_data_size > 0 else ""}
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        {"".join(f'<div class="metric"><strong>{k}:</strong> {v:.4f}</div>' for k, v in card.metrics.items()) if card.metrics else "<p>No metrics provided</p>"}
    </div>

    <div class="section">
        <h2>Ethical Considerations</h2>
        {"<ul>" + "".join(f"<li>{item}</li>" for item in card.ethical_considerations) + "</ul>" if card.ethical_considerations else "<p>None specified</p>"}
    </div>

    <div class="section">
        <h2>Limitations</h2>
        {"<ul>" + "".join(f"<li>{item}</li>" for item in card.limitations) + "</ul>" if card.limitations else "<p>None specified</p>"}
    </div>
</body>
</html>
"""
        return html

    def save_html_report(self, card: ModelCard, filepath: Path | str) -> None:
        """Save HTML report to file.

        Args:
            card: Model card
            filepath: Path to save HTML file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        html = self.generate_html_report(card)

        with open(filepath, "w") as f:
            f.write(html)

        LOGGER.info(f"Saved HTML report to {filepath}")


def generate_model_card(
    model_name: str,
    model_version: str,
    model_type: str,
    description: str,
    metrics: dict[str, float] | None = None,
    **kwargs: Any,
) -> ModelCard:
    """Generate model card (convenience function).

    Args:
        model_name: Model name
        model_version: Model version
        model_type: Type of model
        description: Model description
        metrics: Performance metrics
        **kwargs: Additional card properties

    Returns:
        Generated model card
    """
    generator = ModelCardGenerator()
    card = generator.create_card(
        model_name,
        model_version,
        model_type,
        description,
        **kwargs,
    )

    if metrics:
        generator.add_performance_metrics(card, metrics)

    return card
