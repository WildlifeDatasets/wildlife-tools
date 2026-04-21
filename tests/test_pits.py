"""Tests for per-instance temperature scaling training utilities."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD

from wildlife_tools.train import BasicTrainer, PerInstanceTemperatureScalingLoss


def test_pits_requires_label_counts():
    """The loss should reject forward passes when class counts were not provided."""
    loss = PerInstanceTemperatureScalingLoss()
    logits = torch.randn(2, 3)
    temperature = torch.zeros(2, 1)
    target = torch.tensor([0, 1])

    with pytest.raises(ValueError):
        loss((logits, temperature), target)


def test_pits_returns_finite_loss():
    """The loss should stay finite and propagate gradients for logits and temperature."""
    loss = PerInstanceTemperatureScalingLoss(label_counts=np.array([10, 5, 2]), lambda_weight=0.1)
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.2, -0.5]], requires_grad=True)
    temperature = torch.zeros(2, 1, requires_grad=True)
    target = torch.tensor([0, 1])

    value = loss((logits, temperature), target)
    value.backward()

    assert torch.isfinite(value)
    assert logits.grad is not None
    assert temperature.grad is not None


class DummyDataset:
    """Minimal labeled dataset exposing class counts for trainer integration tests."""

    def __init__(self):
        self.label_counts = np.array([3, 1])

    def __getitem__(self, index):
        return torch.zeros(4), 0

    def __len__(self):
        return 1


class DummyModel(nn.Module):
    """Small model that emits `(logits, temperature)` pairs."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        output = self.linear(x)
        return output[:, :2], output[:, 2:]


def test_basic_trainer_sets_label_counts_for_pits():
    """The basic trainer should forward dataset class counts into the PITS loss."""
    dataset = DummyDataset()
    model = DummyModel()
    objective = PerInstanceTemperatureScalingLoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    trainer = BasicTrainer(
        dataset=dataset,
        model=model,
        objective=objective,
        optimizer=optimizer,
        epochs=1,
        device="cpu",
    )

    assert np.array_equal(trainer.objective.label_counts, dataset.label_counts)
