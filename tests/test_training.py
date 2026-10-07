from __future__ import annotations

import torch

from core.training import EarlyStoppingConfig, train_stfateflow


class ConstantLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def loss(self, batch):
        # Keep a gradient path while exposing a deliberately flat objective.
        loss = self.weight * 0.0 + batch["value"].mean()
        return {"loss": loss}


def _loader():
    while True:
        yield {"value": torch.ones(2)}


def test_early_stopping_stops_on_plateau_and_reports_best_step():
    model = ConstantLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = EarlyStoppingConfig(
        min_steps=4,
        check_interval=2,
        smoothing_window=2,
        patience=2,
        min_relative_improvement=0.01,
    )
    result = train_stfateflow(
        model,
        _loader(),
        optimizer,
        device="cpu",
        max_steps=20,
        early_stopping=config,
        use_amp=False,
        show_progress=False,
    )
    assert result.stopped_early
    assert result.completed_steps == 8
    assert result.best_step == 4
    assert result.stop_reason == "smoothed_loss_plateau"


class ImprovingBelowThresholdModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def loss(self, batch):
        self.calls += 1
        value = 1.0 - self.calls * 0.001
        return {"loss": self.weight * 0.0 + torch.tensor(value)}


def test_best_checkpoint_tracks_small_improvements_without_resetting_patience():
    model = ImprovingBelowThresholdModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = train_stfateflow(
        model,
        _loader(),
        optimizer,
        device="cpu",
        max_steps=20,
        early_stopping=EarlyStoppingConfig(
            min_steps=4,
            check_interval=2,
            smoothing_window=2,
            patience=2,
            min_relative_improvement=0.01,
        ),
        use_amp=False,
        show_progress=False,
    )
    assert result.completed_steps == 8
    assert result.best_step == 8


def test_none_disables_early_stopping_and_defaults_to_fixed_budget():
    model = ConstantLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = train_stfateflow(
        model,
        _loader(),
        optimizer,
        device="cpu",
        max_steps=7,
        early_stopping=None,
        use_amp=False,
        show_progress=False,
    )
    assert not result.stopped_early
    assert result.completed_steps == 7
    assert result.stop_reason == "max_steps_reached"
