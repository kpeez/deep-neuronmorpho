"""Tests for NTXEntLoss."""

import pytest
import torch

from deep_neuronmorpho.engine.ntxent_loss import NTXEntLoss


class TestNTXEntLoss:
    """Class for testing NTXEntLoss."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        torch.manual_seed(0)
        self.loss = NTXEntLoss(temperature=1)
        self.a = torch.randn(32, 128)
        self.self_loss = self.loss(self.a, self.a)

    def test_ntxent_loss_positive(self) -> None:
        """Test NTXEntLoss should always be positive."""
        assert self.self_loss.item() > 0

    def test_ntxent_loss_self_vs_other(self) -> None:
        """Test NTXEntLoss with identical inputs should be less than with distinct inputs."""
        other_loss = self.loss(self.a, torch.randn(32, 128))
        assert self.self_loss.item() < other_loss.item()

    def test_ntxent_loss_high_temp(self) -> None:
        """Test NTXEntLoss with high temperature should be larger."""
        loss_high_temp = NTXEntLoss(temperature=1000.0)
        self_loss_high = loss_high_temp(self.a, self.a)
        assert self_loss_high.item() > 0
        assert self_loss_high.item() > self.self_loss.item()

    def test_ntxent_loss_low_temp(self) -> None:
        """Test NTXEntLoss with low temperature should be less."""
        loss_low_temp = NTXEntLoss(temperature=0.1)
        self_loss_low = loss_low_temp(self.a, self.a)
        assert self_loss_low.item() > 0
        assert self_loss_low.item() < self.self_loss.item()
