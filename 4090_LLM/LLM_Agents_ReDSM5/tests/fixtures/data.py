"""
Synthetic data generation for testing ReDSM5 models.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

# DSM-5 depression symptom labels
LABEL_NAMES = [
    "depressed_mood",
    "diminished_interest",
    "weight_appetite_change",
    "sleep_disturbance",
    "psychomotor",
    "fatigue",
    "worthlessness_guilt",
    "concentration_indecision",
    "suicidality"
]

# Keywords associated with each symptom for realistic text generation
KEYWORDS = {
    "depressed_mood": ["sad", "down", "depressed", "hopeless", "empty", "miserable"],
    "diminished_interest": ["no interest", "don't enjoy", "lost pleasure", "anhedonia", "unmotivated"],
    "weight_appetite_change": ["weight loss", "weight gain", "appetite", "eating", "hungry", "full"],
    "sleep_disturbance": ["can't sleep", "insomnia", "sleep too much", "tired", "restless nights"],
    "psychomotor": ["agitated", "restless", "slowed down", "can't sit still", "moving slowly"],
    "fatigue": ["exhausted", "no energy", "tired", "fatigue", "drained", "weary"],
    "worthlessness_guilt": ["worthless", "guilty", "failure", "inadequate", "ashamed"],
    "concentration_indecision": ["can't focus", "can't decide", "distracted", "forgetful", "confused"],
    "suicidality": ["want to die", "suicide", "end it all", "not worth living", "self-harm"]
}


def generate_synthetic_sample(
    idx: int,
    label_names: List[str] = LABEL_NAMES,
    seed: int = 42
) -> Dict:
    """
    Generate a single synthetic sample with text and binary labels.

    Args:
        idx: Sample index for reproducibility
        label_names: List of label names
        seed: Random seed base

    Returns:
        Dictionary with 'text' and label columns
    """
    random.seed(seed + idx)

    # Generate random binary labels (20% positive rate per label)
    labels = {label: random.random() < 0.2 for label in label_names}

    # Generate text based on positive labels
    text_parts = []
    for label, is_positive in labels.items():
        if is_positive and label in KEYWORDS:
            keyword = random.choice(KEYWORDS[label])
            text_parts.append(f"I feel {keyword}.")

    # Add some filler text if no symptoms
    if not text_parts:
        text_parts = ["I'm posting to share my thoughts.", "Just checking in with everyone."]

    # Combine into a single text
    text = " ".join(text_parts)

    # Return sample with text and integer labels
    sample = {"text": text}
    sample.update({label: int(value) for label, value in labels.items()})

    return sample


def generate_synthetic_dataset(
    output_dir: Path,
    num_samples: int = 100,
    seed: int = 42,
    label_names: List[str] = LABEL_NAMES
) -> None:
    """
    Generate a complete synthetic dataset with train/dev/test splits.

    Args:
        output_dir: Directory to write JSONL files
        num_samples: Total number of samples to generate
        seed: Random seed
        label_names: List of label names
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    samples = [generate_synthetic_sample(i, label_names, seed) for i in range(num_samples)]

    # Split into train (70%), dev (15%), test (15%)
    train_size = int(0.7 * num_samples)
    dev_size = int(0.15 * num_samples)

    train_samples = samples[:train_size]
    dev_samples = samples[train_size:train_size + dev_size]
    test_samples = samples[train_size + dev_size:]

    # Write JSONL files
    for split_name, split_samples in [
        ("train", train_samples),
        ("dev", dev_samples),
        ("test", test_samples)
    ]:
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w") as f:
            for sample in split_samples:
                f.write(json.dumps(sample) + "\n")


def generate_edge_case_samples() -> List[Dict]:
    """Generate edge case samples for robustness testing."""
    return [
        # Empty text
        {"text": "", **{label: 0 for label in LABEL_NAMES}},

        # Very long text (repeated)
        {"text": "This is a test sentence. " * 1000, **{label: 0 for label in LABEL_NAMES}},

        # All labels positive
        {"text": "I feel terrible in every way.", **{label: 1 for label in LABEL_NAMES}},

        # All labels negative
        {"text": "I feel great today!", **{label: 0 for label in LABEL_NAMES}},

        # Special characters
        {"text": "I feel @#$%^&* terrible!!!", **{label: 1 for label in ["depressed_mood"]}},
    ]
