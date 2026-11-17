"""Monte Carlo dropout with mean-only or heteroscedastic output."""

from __future__ import annotations

import copy
import numpy as np
import torch

from ._common import (
    DEFAULT_LEARNING_RATE,
    TORCH_DTYPE,
    GaussianMLP,
    TargetStandardizer,
    as_2d,
    gaussian_nll,
    set_seed,
)


class MCDropoutModel:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 n_samples: int = 5,
                 dropout_probability: float = 0.1,
                 heteroscedastic: bool = True,
                 hidden_size: int | None = None,
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 max_epochs: int = 400, batch_size: int = 32, patience: int = 5,
                 warmup_epochs: int = 50, min_delta: float = 1e-4,
                 random_seed: int = 0):
        self.n_samples, self.dropout_probability = n_samples, dropout_probability
        self.heteroscedastic = heteroscedastic
        self.hidden_size = hidden_size
        self.learning_rate, self.max_epochs = learning_rate, max_epochs
        self.batch_size, self.patience = batch_size, patience
        self.warmup_epochs = warmup_epochs
        self.min_delta = min_delta
        self.random_seed = random_seed
        self.fit(X_train, y_train)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MCDropoutModel":
        X = as_2d(X_train)
        y_original = np.asarray(y_train).reshape(-1)
        if len(X) != len(y_original):
            raise ValueError("X_train and y_train must have equal length")
        if (
            self.n_samples <= 0
            or self.max_epochs <= 0
            or self.batch_size <= 0
            or self.patience <= 0
        ):
            raise ValueError(
                "n_samples, max_epochs, batch_size, and patience must be positive"
            )
        self.target_scaler = TargetStandardizer.fit(y_original)
        y = self.target_scaler.transform(y_original)
        X_tensor = torch.as_tensor(X, dtype=TORCH_DTYPE)
        y_tensor = torch.as_tensor(y, dtype=TORCH_DTYPE)
        indices = torch.arange(len(X_tensor))
        generator = torch.Generator().manual_seed(self.random_seed)

        set_seed(self.random_seed)
        self.model = GaussianMLP(
            X.shape[1],
            hidden_size=self.hidden_size,
            dropout=self.dropout_probability,
            heteroscedastic=self.heteroscedastic,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        self.best_epoch_ = 0
        stale = 0
        first_monitored_epoch = min(
            max(1, self.warmup_epochs), self.max_epochs
        )
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            shuffled = indices[torch.randperm(len(indices), generator=generator)]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                optimizer.zero_grad()
                mean, variance = self.model(X_tensor[batch])
                loss = (
                    gaussian_nll(mean, variance, y_tensor[batch])
                    if self.heteroscedastic
                    else torch.nn.functional.mse_loss(mean, y_tensor[batch])
                )
                loss.backward()
                optimizer.step()

            if epoch < first_monitored_epoch:
                continue

            # Preserve the historical checkpoint-selection procedure: disable
            # dropout and evaluate one deterministic pass. The only change is
            # that the complete training set replaces the former validation set.
            self.model.eval()
            with torch.no_grad():
                mean, variance = self.model(X_tensor)
            training_loss = float(
                gaussian_nll(mean, variance, y_tensor)
                if self.heteroscedastic
                else torch.nn.functional.mse_loss(mean, y_tensor)
            )
            if training_loss < best_loss - self.min_delta:
                best_loss = training_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.optimization_epochs_ = epoch
        self.training_objective_ = best_loss
        if self.heteroscedastic:
            self.training_nll_ = best_loss
        else:
            self.training_mse_ = best_loss
        return self

    def _standardized_components(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.train()  # dropout intentionally remains active
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(self.random_seed + 10_000)
        means, variances = [], []
        with torch.no_grad():
            for _ in range(self.n_samples):
                mean, variance = self.model(X)
                means.append(mean)
                variances.append(variance)
        torch.random.set_rng_state(generator_state)
        return torch.stack(means), torch.stack(variances)

    def predict_components(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = torch.as_tensor(as_2d(X_test), dtype=TORCH_DTYPE)
        means, variances = self._standardized_components(X)
        return (
            means.numpy() * self.target_scaler.std + self.target_scaler.mean,
            variances.numpy() * self.target_scaler.std**2,
        )

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, variances = self.predict_components(X_test)
        mean = means.mean(axis=0)
        variance = (variances + means**2).mean(axis=0) - mean**2
        return mean, np.maximum(variance, 1e-12)
