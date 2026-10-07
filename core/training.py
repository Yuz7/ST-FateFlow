"""Training utilities for ST-FateFlow.

The early-stopping criterion deliberately uses only the stochastic flow-matching
training objective.  In interpolation experiments, the held-out time point must
not be used to decide when to stop training.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for smoothed-loss early stopping.

    A check is an improvement only when the rolling mean decreases by at least
    ``min_relative_improvement`` relative to the best previous check.
    """

    min_steps: int = 500
    check_interval: int = 50
    smoothing_window: int = 100
    patience: int = 4
    min_relative_improvement: float = 0.005
    restore_best: bool = True

    def validate(self, max_steps: int) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0 < self.min_steps <= max_steps:
            raise ValueError("min_steps must be in [1, max_steps]")
        if self.check_interval <= 0 or self.smoothing_window <= 0 or self.patience <= 0:
            raise ValueError("check_interval, smoothing_window, and patience must be positive")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("min_relative_improvement must be in [0, 1)")


@dataclass
class TrainingResult:
    requested_steps: int
    completed_steps: int
    stopped_early: bool
    stop_reason: str
    best_step: int
    best_smoothed_loss: float
    loss_history: list[float]
    early_stopping: dict[str, Any] | None

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("loss_history")
        return payload


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_stfateflow(
    model: nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device,
    max_steps: int = 1000,
    early_stopping: EarlyStoppingConfig | None = EarlyStoppingConfig(),
    max_grad_norm: float = 1.0,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool | None = None,
    scaler: Any | None = None,
    show_progress: bool = True,
) -> TrainingResult:
    """Train ST-FateFlow with optional leakage-free early stopping.

    Passing ``early_stopping=None`` disables early stopping and always performs
    exactly ``max_steps`` optimizer updates (1,000 by default).
    """

    device = torch.device(device)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if early_stopping is not None:
        early_stopping.validate(max_steps)
    if use_amp is None:
        use_amp = device.type == "cuda"

    model.train()
    iterator = iter(train_loader)
    losses: list[float] = []
    rolling_losses: deque[float] = deque(
        maxlen=early_stopping.smoothing_window if early_stopping is not None else 1
    )
    best_metric = float("inf")
    improvement_reference = float("inf")
    best_step = 0
    best_state: dict[str, Any] | None = None
    checks_without_improvement = 0
    stopped_early = False
    stop_reason = "max_steps_reached"

    progress = tqdm(range(1, max_steps + 1), desc="Training ST-FateFlow", disable=not show_progress)
    for step in progress:
        batch = _move_batch(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss_dict = model.loss(batch)
            loss = loss_dict["loss"]

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}: {loss.item()}")

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        rolling_losses.append(loss_value)

        if early_stopping is None:
            continue
        should_check = (
            step >= early_stopping.min_steps
            and step % early_stopping.check_interval == 0
            and len(rolling_losses) == early_stopping.smoothing_window
        )
        if not should_check:
            continue

        metric = float(np.mean(rolling_losses))
        if metric < best_metric:
            best_metric = metric
            best_step = step
            if early_stopping.restore_best:
                best_state = _cpu_state_dict(model)

        threshold = improvement_reference * (1.0 - early_stopping.min_relative_improvement)
        if metric < threshold:
            improvement_reference = metric
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        progress.set_postfix(smoothed_loss=f"{metric:.5g}", patience=checks_without_improvement)
        if checks_without_improvement >= early_stopping.patience and step < max_steps:
            stopped_early = True
            stop_reason = "smoothed_loss_plateau"
            break

    completed_steps = len(losses)
    if early_stopping is not None and best_step == 0:
        best_step = completed_steps
        best_metric = float(np.mean(losses[-early_stopping.smoothing_window :]))
    if early_stopping is not None and early_stopping.restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(
        requested_steps=max_steps,
        completed_steps=completed_steps,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        best_step=best_step,
        best_smoothed_loss=best_metric,
        loss_history=losses,
        early_stopping=asdict(early_stopping) if early_stopping is not None else None,
    )
