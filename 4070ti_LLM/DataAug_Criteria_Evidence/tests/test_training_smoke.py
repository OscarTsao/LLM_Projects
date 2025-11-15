"""Smoke tests for training pipeline."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from psy_agents_noaug.training.evaluate import Evaluator
from psy_agents_noaug.training.train_loop import Trainer


@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    from torch.utils.data import Dataset

    batch_size = 4
    seq_len = 32
    num_samples = 16

    class DictDataset(Dataset):
        """Dataset that returns dictionaries instead of tuples."""

        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    input_ids = torch.randint(0, 1000, (num_samples, seq_len))
    attention_mask = torch.ones(num_samples, seq_len)
    labels = torch.randint(0, 3, (num_samples,))

    dataset = DictDataset(input_ids, attention_mask, labels)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


@pytest.fixture
def dummy_model():
    """Create dummy model for testing."""

    # Note: This is a minimal mock, not using actual BERT
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 64
            self.embedding = nn.Embedding(1000, 64)

        def forward(self, input_ids, attention_mask):
            # Simple mean pooling
            embeds = self.embedding(input_ids)
            return embeds.mean(dim=1)

    encoder = DummyEncoder()

    class DummyModel(nn.Module):
        def __init__(self, encoder, num_classes):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(encoder.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask):
            encodings = self.encoder(input_ids, attention_mask)
            return self.classifier(encodings)

    return DummyModel(encoder, num_classes=3)


def test_trainer_initialization(dummy_model, dummy_data):
    """Test that trainer can be initialized."""
    train_loader, val_loader = dummy_data

    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    trainer = Trainer(
        model=dummy_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=2,
        patience=1,
    )

    assert trainer is not None, "Trainer should be initialized"


def test_trainer_can_run_epoch(dummy_model, dummy_data):
    """Test that trainer can run one epoch."""
    train_loader, val_loader = dummy_data

    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    trainer = Trainer(
        model=dummy_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=1,
    )

    metrics = trainer.train_epoch(0)

    assert "train_loss" in metrics, "Should return train loss"
    assert "train_accuracy" in metrics, "Should return train accuracy"
    assert metrics["train_loss"] > 0, "Loss should be positive"


def test_evaluator_can_evaluate(dummy_model, dummy_data):
    """Test that evaluator can run evaluation."""
    _, val_loader = dummy_data

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    evaluator = Evaluator(
        model=dummy_model,
        device=device,
        criterion=criterion,
    )

    metrics = evaluator.evaluate(val_loader)

    assert "accuracy" in metrics, "Should return accuracy"
    assert "loss" in metrics, "Should return loss"
    assert "f1_macro" in metrics, "Should return F1 score"


def test_evaluator_can_predict(dummy_model, dummy_data):
    """Test that evaluator can make predictions."""
    _, val_loader = dummy_data

    device = torch.device("cpu")

    evaluator = Evaluator(
        model=dummy_model,
        device=device,
    )

    predictions = evaluator.predict(val_loader)

    assert len(predictions) == 16, "Should return predictions for all samples"
    assert (
        predictions.dtype == int or predictions.dtype == torch.long
    ), "Predictions should be integers"
