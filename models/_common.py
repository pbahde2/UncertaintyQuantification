"""Shared utilities for the trained uncertainty models."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


# Use one numerical precision across GP and neural-network methods so runtime and
# predictive comparisons are not confounded by float32 versus float64 arithmetic.
NUMPY_DTYPE = np.float64
TORCH_DTYPE = torch.float64
DEFAULT_LEARNING_RATE = 0.01


class TrainingObjectiveStoppingCriterion:
    """Patience stopping for a scalar full-training objective."""

    def __init__(
        self,
        *,
        first_monitored_step: int,
        patience: int,
        min_delta: float,
    ) -> None:
        if first_monitored_step <= 0 or patience <= 0 or min_delta < 0:
            raise ValueError(
                "first_monitored_step and patience must be positive and "
                "min_delta must be non-negative"
            )
        self.first_monitored_step = first_monitored_step
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.best_loss = float("inf")
        self.best_step = 0
        self.stale_steps = 0
        self.improved = False
        self.converged = False

    def __call__(self, fvals: torch.Tensor) -> bool:
        if fvals.numel() != 1:
            raise ValueError("Stopping requires a scalar training objective")
        self.step += 1
        self.improved = False
        if self.step < self.first_monitored_step:
            return False

        loss = float(fvals.detach())
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_step = self.step
            self.stale_steps = 0
            self.improved = True
        else:
            self.stale_steps += 1
        self.converged = self.stale_steps >= self.patience
        return self.converged


def as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=NUMPY_DTYPE)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TargetStandardizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, y: np.ndarray) -> "TargetStandardizer":
        y = np.asarray(y, dtype=NUMPY_DTYPE).reshape(-1)
        std = float(np.std(y))
        return cls(float(np.mean(y)), std if std > 1e-8 else 1.0)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=NUMPY_DTYPE) - self.mean) / self.std


def resolve_hidden_size(input_dim: int, hidden_size: int | None) -> int:
    """Return an explicit hidden width or the benchmark's dimension-based default."""
    width = max(2, 2 * input_dim) if hidden_size is None else int(hidden_size)
    if width <= 0:
        raise ValueError("hidden_size must be positive")
    return width


class GaussianMLP(nn.Module):
    """Two-hidden-layer MLP with an optional heteroscedastic variance head."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int | None = None,
        dropout: float = 0.0,
        heteroscedastic: bool = True,
    ) -> None:
        super().__init__()
        width = resolve_hidden_size(input_dim, hidden_size)
        self.heteroscedastic = heteroscedastic
        self.hidden1 = nn.Linear(input_dim, width, dtype=TORCH_DTYPE)
        self.hidden2 = nn.Linear(width, width, dtype=TORCH_DTYPE)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(
            width, 2 if heteroscedastic else 1, dtype=TORCH_DTYPE
        )

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.dropout(torch.relu(self.hidden1(X)))
        hidden = self.dropout(torch.relu(self.hidden2(hidden)))
        output = self.head(hidden)
        if not self.heteroscedastic:
            mean = output.squeeze(-1)
            return mean, torch.zeros_like(mean)
        mean, raw_variance = output.chunk(2, dim=-1)
        variance = torch.nn.functional.softplus(raw_variance) + 1e-6
        return mean.squeeze(-1), variance.squeeze(-1)


def gaussian_nll(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log(variance) + (target - mean).square() / variance).mean()


def moment_match_gaussian_components(
    means: torch.Tensor, variances: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Moment-match Gaussian components exactly as the public predictors do."""
    mean = means.mean(dim=0)
    variance = (variances + means.square()).mean(dim=0) - mean.square()
    return mean, variance.clamp_min(1e-12)
