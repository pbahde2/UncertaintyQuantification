"""Fixed-width MLP variants for the one-dimensional synthetic experiment."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ._common import (
    TORCH_DTYPE,
    TargetStandardizer,
    as_2d,
    set_seed,
    train_gaussian_network,
)
from .deep_ensemble import DeepEnsembleModel
from .mc_dropout import MCDropoutModel


class _FixedWidthGaussianMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_size, dtype=TORCH_DTYPE)
        self.hidden2 = nn.Linear(hidden_size, hidden_size, dtype=TORCH_DTYPE)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 2, dtype=TORCH_DTYPE)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.dropout(torch.relu(self.hidden1(X)))
        hidden = self.dropout(torch.relu(self.hidden2(hidden)))
        mean, raw_variance = self.head(hidden).chunk(2, dim=-1)
        variance = torch.nn.functional.softplus(raw_variance) + 1e-6
        return mean.squeeze(-1), variance.squeeze(-1)


class FixedWidthDeepEnsembleModel(DeepEnsembleModel):
    """Deep Ensemble whose member networks have an explicit hidden size."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hidden_size: int,
        **kwargs: object,
    ) -> None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        super().__init__(X_train, y_train, **kwargs)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "FixedWidthDeepEnsembleModel":
        X = as_2d(X_train)
        self.target_scaler = TargetStandardizer.fit(y_train)
        y = self.target_scaler.transform(y_train)
        y_val_scaled = self.target_scaler.transform(y_val) if y_val is not None else None
        self.members = []
        self.selected_epochs_ = []
        for member_idx in range(self.n_members):
            seed = self.random_seed + member_idx
            set_seed(seed)
            member = _FixedWidthGaussianMLP(X.shape[1], self.hidden_size)
            best_epoch = train_gaussian_network(
                member,
                X,
                y,
                learning_rate=self.learning_rate,
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                patience=self.patience,
                warmup_epochs=self.warmup_epochs,
                min_delta=self.min_delta,
                validation_fraction=self.validation_fraction,
                seed=seed,
                X_val=X_val,
                y_val=y_val_scaled,
            )
            member.eval()
            self.members.append(member)
            self.selected_epochs_.append(best_epoch)
        return self


class FixedWidthMCDropoutModel(MCDropoutModel):
    """MC Dropout model whose hidden layers have an explicit size."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hidden_size: int,
        **kwargs: object,
    ) -> None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        super().__init__(X_train, y_train, **kwargs)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "FixedWidthMCDropoutModel":
        X = as_2d(X_train)
        self.target_scaler = TargetStandardizer.fit(y_train)
        y = self.target_scaler.transform(y_train)
        y_val_scaled = self.target_scaler.transform(y_val) if y_val is not None else None
        set_seed(self.random_seed)
        self.model = _FixedWidthGaussianMLP(
            X.shape[1], self.hidden_size, dropout=self.dropout_probability
        )
        self.best_epoch_ = train_gaussian_network(
            self.model,
            X,
            y,
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            patience=self.patience,
            warmup_epochs=self.warmup_epochs,
            min_delta=self.min_delta,
            validation_fraction=self.validation_fraction,
            seed=self.random_seed,
            X_val=X_val,
            y_val=y_val_scaled,
        )
        return self
