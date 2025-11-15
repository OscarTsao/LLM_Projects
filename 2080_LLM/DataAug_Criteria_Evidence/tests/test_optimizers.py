"""Tests for optimizer factory (SUPERMAX Phase 4)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from psy_agents_noaug.training.optimizers import (
    check_optimizer_available,
    create_optimizer,
    get_optimizer_info,
    list_available_optimizers,
)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Linear(10, 2)


class TestOptimizerCreation:
    """Test optimizer creation for all supported types."""

    def test_create_adamw(self, simple_model):
        """Test AdamW optimizer creation."""
        optimizer = create_optimizer(
            name="adamw",
            model_parameters=simple_model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
        )

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.defaults["weight_decay"] == 0.01

    def test_create_adam(self, simple_model):
        """Test Adam optimizer creation."""
        optimizer = create_optimizer(
            name="adam",
            model_parameters=simple_model.parameters(),
            lr=5e-4,
            weight_decay=0.0,
        )

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 5e-4

    def test_create_adafactor(self, simple_model):
        """Test Adafactor optimizer creation."""
        optimizer = create_optimizer(
            name="adafactor",
            model_parameters=simple_model.parameters(),
            lr=1e-2,
            weight_decay=0.0,
        )

        # Adafactor should work (transformers is a core dependency)
        assert optimizer is not None
        # Type will be Adafactor if available
        assert hasattr(optimizer, "step")

    def test_create_lion(self, simple_model):
        """Test Lion optimizer creation (with fallback)."""
        optimizer = create_optimizer(
            name="lion",
            model_parameters=simple_model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )

        # Should create either Lion or fallback to AdamW
        assert optimizer is not None
        assert hasattr(optimizer, "step")

    def test_create_lamb(self, simple_model):
        """Test LAMB optimizer creation (with fallback)."""
        optimizer = create_optimizer(
            name="lamb",
            model_parameters=simple_model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
        )

        # Should create either LAMB or fallback to AdamW
        assert optimizer is not None
        assert hasattr(optimizer, "step")

    def test_create_adamw_8bit(self, simple_model):
        """Test AdamW-8bit optimizer creation (with fallback)."""
        optimizer = create_optimizer(
            name="adamw_8bit",
            model_parameters=simple_model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
        )

        # Should create either AdamW8bit or fallback to AdamW
        assert optimizer is not None
        assert hasattr(optimizer, "step")

    def test_case_insensitive_names(self, simple_model):
        """Test optimizer names are case-insensitive."""
        opt1 = create_optimizer("AdamW", simple_model.parameters(), lr=1e-3)
        opt2 = create_optimizer("ADAMW", simple_model.parameters(), lr=1e-3)
        opt3 = create_optimizer("adamw", simple_model.parameters(), lr=1e-3)

        assert opt1.__class__.__name__ == opt2.__class__.__name__
        assert opt2.__class__.__name__ == opt3.__class__.__name__

    def test_unknown_optimizer_raises(self, simple_model):
        """Test that unknown optimizer name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(
                name="unknown_opt",
                model_parameters=simple_model.parameters(),
                lr=1e-3,
            )


class TestOptimizerParameters:
    """Test optimizer parameter handling."""

    def test_learning_rate_propagation(self, simple_model):
        """Test that learning rate is correctly set."""
        test_lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

        for lr in test_lrs:
            optimizer = create_optimizer(
                "adamw", simple_model.parameters(), lr=lr, weight_decay=0.0
            )
            assert optimizer.defaults["lr"] == lr

    def test_weight_decay_propagation(self, simple_model):
        """Test that weight decay is correctly set."""
        test_wds = [0.0, 0.001, 0.01, 0.1]

        for wd in test_wds:
            optimizer = create_optimizer(
                "adamw", simple_model.parameters(), lr=1e-3, weight_decay=wd
            )
            assert optimizer.defaults["weight_decay"] == wd

    def test_custom_betas(self, simple_model):
        """Test that custom beta parameters are respected."""
        optimizer = create_optimizer(
            "adamw",
            simple_model.parameters(),
            lr=1e-3,
            betas=(0.8, 0.95),
        )

        assert optimizer.defaults["betas"] == (0.8, 0.95)

    def test_custom_eps(self, simple_model):
        """Test that custom epsilon parameter is respected."""
        optimizer = create_optimizer(
            "adamw",
            simple_model.parameters(),
            lr=1e-3,
            eps=1e-6,
        )

        assert optimizer.defaults["eps"] == 1e-6


class TestOptimizerFunctionality:
    """Test optimizer training functionality."""

    def test_optimizer_step_works(self, simple_model):
        """Test that optimizer can perform a training step."""
        optimizer = create_optimizer(
            "adamw", simple_model.parameters(), lr=1e-3, weight_decay=0.01
        )

        # Create dummy input and target
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        # Forward pass
        output = simple_model(x)
        loss = nn.CrossEntropyLoss()(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step should work
        optimizer.step()

        # Loss should change on next iteration
        output2 = simple_model(x)
        loss2 = nn.CrossEntropyLoss()(output2, y)
        assert loss.item() != loss2.item()

    @pytest.mark.parametrize(
        "optimizer_name", ["adamw", "adam", "adafactor", "lion", "lamb", "adamw_8bit"]
    )
    def test_all_optimizers_can_train(self, simple_model, optimizer_name):
        """Test that all optimizers can perform training steps."""
        optimizer = create_optimizer(
            optimizer_name, simple_model.parameters(), lr=1e-3, weight_decay=0.01
        )

        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        initial_params = [p.clone() for p in simple_model.parameters()]

        # Train for a few steps
        for _ in range(3):
            output = simple_model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Parameters should have changed
        for initial, current in zip(
            initial_params, simple_model.parameters(), strict=False
        ):
            assert not torch.allclose(initial, current)


class TestOptimizerInfo:
    """Test optimizer metadata functions."""

    def test_get_optimizer_info_all_supported(self):
        """Test that info is available for all supported optimizers."""
        supported = ["adamw", "adam", "adafactor", "lion", "lamb", "adamw_8bit"]

        for name in supported:
            info = get_optimizer_info(name)
            assert info is not None
            assert "name" in info
            assert "memory_efficient" in info
            assert "recommended_lr" in info
            assert "requires" in info
            assert "notes" in info

    def test_list_available_optimizers(self):
        """Test that list of available optimizers is correct."""
        optimizers = list_available_optimizers()

        assert "adamw" in optimizers
        assert "adam" in optimizers
        assert "adafactor" in optimizers
        assert "lion" in optimizers
        assert "lamb" in optimizers
        assert "adamw_8bit" in optimizers
        assert len(optimizers) == 6

    def test_check_optimizer_available(self):
        """Test optimizer availability checking."""
        # Standard optimizers should always be available
        available, error = check_optimizer_available("adamw")
        assert available is True
        assert error is None

        available, error = check_optimizer_available("adam")
        assert available is True
        assert error is None

        # Unknown optimizer should return false
        available, error = check_optimizer_available("unknown")
        assert available is False
        assert error is not None

    def test_optimizer_info_memory_efficiency(self):
        """Test that memory efficiency flags are correct."""
        # Standard optimizers are not memory-efficient
        info_adamw = get_optimizer_info("adamw")
        assert info_adamw["memory_efficient"] is False

        # Memory-efficient optimizers
        memory_efficient = ["adafactor", "lion", "adamw_8bit"]
        for name in memory_efficient:
            info = get_optimizer_info(name)
            assert info["memory_efficient"] is True

    def test_optimizer_info_lr_recommendations(self):
        """Test that learning rate recommendations are reasonable."""
        info_adamw = get_optimizer_info("adamw")
        lr_low, lr_high = info_adamw["recommended_lr"]

        assert lr_low > 0
        assert lr_high > lr_low
        assert lr_low >= 1e-6
        assert lr_high <= 1e-1


class TestOptimizerHPOIntegration:
    """Test optimizer integration with HPO system."""

    def test_hpo_search_space_includes_all_optimizers(self):
        """Test that HPO search space includes all 6 optimizers."""
        import optuna

        from psy_agents_noaug.hpo import SearchSpace

        space = SearchSpace("criteria")
        study = optuna.create_study()
        trial = study.ask()

        params = space.sample(trial)

        # Optimizer name should be sampled
        assert "optim.name" in params
        assert params["optim.name"] in [
            "adamw",
            "adam",
            "adafactor",
            "lion",
            "lamb",
            "adamw_8bit",
        ]

    def test_hpo_can_sample_all_optimizers(self):
        """Test that HPO can sample all 6 optimizer types."""
        import optuna

        from psy_agents_noaug.hpo import SearchSpace, SpaceConstraints

        space = SearchSpace("criteria")
        optimizer_names = [
            "adamw",
            "adam",
            "adafactor",
            "lion",
            "lamb",
            "adamw_8bit",
        ]

        for opt_name in optimizer_names:
            study = optuna.create_study()
            trial = study.ask()
            constraints = SpaceConstraints(categorical={"optim.name": [opt_name]})
            params = space.sample(trial, constraints)

            assert params["optim.name"] == opt_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
