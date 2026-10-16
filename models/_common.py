"""Shared utilities for the trained uncertainty models."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
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
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        std = float(np.std(y))
        return cls(float(np.mean(y)), std if std > 1e-8 else 1.0)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float32) - self.mean) / self.std


class GaussianMLP(nn.Module):
    """One-hidden-layer MLP producing a Gaussian mean and variance."""

    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        width = max(2, 2 * input_dim)
        self.hidden = nn.Linear(input_dim, width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 2)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.dropout(torch.relu(self.hidden(X)))
        mean, raw_variance = self.head(hidden).chunk(2, dim=-1)
        variance = torch.nn.functional.softplus(raw_variance) + 1e-6
        return mean.squeeze(-1), variance.squeeze(-1)


def gaussian_nll(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log(variance) + (target - mean).square() / variance).mean()


def train_gaussian_network(
    model: GaussianMLP,
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    max_epochs: int,
    batch_size: int,
    patience: int,
    warmup_epochs: int,
    min_delta: float,
    validation_fraction: float,
    seed: int,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    fixed_epochs: int | None = None,
) -> int:
    """Train using Gaussian NLL and retain the best validation checkpoint."""
    set_seed(seed)
    X_tensor = torch.as_tensor(as_2d(X), dtype=torch.float32)
    y_tensor = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1))
    generator = torch.Generator().manual_seed(seed)
    # During final retraining, use every observation and disable validation stopping.
    if fixed_epochs is not None:
        if fixed_epochs <= 0:
            raise ValueError("fixed_epochs must be positive")
        train_idx = torch.arange(len(X_tensor))
        X_val_tensor = y_val_tensor = None
        epochs = fixed_epochs
    elif (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must either both be provided or both be omitted")
    elif X_val is not None and len(X_val):
        train_idx = torch.arange(len(X_tensor))
        X_val_tensor = torch.as_tensor(as_2d(X_val), dtype=torch.float32)
        y_val_tensor = torch.as_tensor(np.asarray(y_val, dtype=np.float32).reshape(-1))
        epochs = max_epochs
    else:
        order = torch.randperm(len(X_tensor), generator=generator)
        n_val = (
            min(max(1, int(round(len(X_tensor) * validation_fraction))), len(X_tensor) - 1)
            if len(X_tensor) > 1 and validation_fraction > 0.0 else 0
        )
        val_idx, train_idx = order[:n_val], order[n_val:]
        if not len(train_idx):
            train_idx = order
        X_val_tensor = X_tensor[val_idx] if n_val else X_tensor[train_idx]
        y_val_tensor = y_tensor[val_idx] if n_val else y_tensor[train_idx]
        epochs = max_epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    best_epoch = 0
    first_monitored_epoch = min(max(1, warmup_epochs), epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = train_idx[torch.randperm(len(train_idx), generator=generator)]
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start : start + batch_size]
            optimizer.zero_grad()
            mean, variance = model(X_tensor[idx])
            loss = gaussian_nll(mean, variance, y_tensor[idx])
            loss.backward()
            optimizer.step()

        # Fixed-duration retraining deliberately performs no validation checks.
        if fixed_epochs is not None:
            continue
        if epoch < first_monitored_epoch:
            continue
        model.eval()
        with torch.no_grad():
            mean, variance = model(X_val_tensor)
            val_loss = float(gaussian_nll(mean, variance, y_val_tensor))
        # An epoch only counts as improved when validation NLL drops by min_delta.
        if val_loss < best_loss - min_delta:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            best_epoch = epoch
        else:
            stale += 1
            # Stop after the configured number of consecutive non-improving epochs.
            if stale >= patience:
                break
    if fixed_epochs is not None:
        return fixed_epochs
    model.load_state_dict(best_state)
    return best_epoch
