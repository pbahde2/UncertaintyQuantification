"""Monte Carlo dropout with a heteroscedastic Gaussian output head."""

from __future__ import annotations

import numpy as np
import torch

from ._common import GaussianMLP, TargetStandardizer, as_2d, set_seed, train_gaussian_network


class MCDropoutModel:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_samples: int = 5,
                 dropout_probability: float = 0.1, learning_rate: float = 1e-3,
                 max_epochs: int = 400, batch_size: int = 32, patience: int = 5,
                 warmup_epochs: int = 50, min_delta: float = 1e-4,
                 validation_fraction: float = 0.2, random_seed: int = 0,
                 X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                 **_: object):
        self.n_samples, self.dropout_probability = n_samples, dropout_probability
        self.learning_rate, self.max_epochs = learning_rate, max_epochs
        self.batch_size, self.patience = batch_size, patience
        self.warmup_epochs, self.min_delta = warmup_epochs, min_delta
        self.validation_fraction, self.random_seed = validation_fraction, random_seed
        self.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "MCDropoutModel":
        X = as_2d(X_train)
        self.target_scaler = TargetStandardizer.fit(y_train)
        y = self.target_scaler.transform(y_train)
        y_val_scaled = self.target_scaler.transform(y_val) if y_val is not None else None
        set_seed(self.random_seed)
        self.model = GaussianMLP(X.shape[1], dropout=self.dropout_probability)
        self.best_epoch_ = train_gaussian_network(
            self.model, X, y, learning_rate=self.learning_rate, max_epochs=self.max_epochs,
            batch_size=self.batch_size, patience=self.patience,
            warmup_epochs=self.warmup_epochs, min_delta=self.min_delta,
            validation_fraction=self.validation_fraction, seed=self.random_seed,
            X_val=X_val, y_val=y_val_scaled,
        )
        return self

    def refit_full(self, X_train: np.ndarray, y_train: np.ndarray) -> "MCDropoutModel":
        """Reinitialize and train on all labels for the selected duration."""
        # Preserve the validation-selected duration before replacing the selection model.
        epochs = self.best_epoch_
        X = as_2d(X_train)
        self.target_scaler = TargetStandardizer.fit(y_train)
        y = self.target_scaler.transform(y_train)
        set_seed(self.random_seed)
        # Create a fresh network so validation observations enter only the final full-data fit.
        self.model = GaussianMLP(X.shape[1], dropout=self.dropout_probability)
        train_gaussian_network(
            self.model, X, y, learning_rate=self.learning_rate, max_epochs=epochs,
            batch_size=self.batch_size, patience=self.patience,
            warmup_epochs=self.warmup_epochs, min_delta=self.min_delta,
            validation_fraction=0.0, seed=self.random_seed, fixed_epochs=epochs,
        )
        return self

    def predict_components(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = torch.as_tensor(as_2d(X_test), dtype=torch.float32)
        self.model.train()  # dropout intentionally remains active
        means, variances = [], []
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(self.random_seed + 10_000)
        with torch.no_grad():
            for _ in range(self.n_samples):
                mean, variance = self.model(X)
                means.append(mean.numpy() * self.target_scaler.std + self.target_scaler.mean)
                variances.append(variance.numpy() * self.target_scaler.std**2)
        torch.random.set_rng_state(generator_state)
        return np.stack(means), np.stack(variances)

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, variances = self.predict_components(X_test)
        mean = means.mean(axis=0)
        variance = (variances + means**2).mean(axis=0) - mean**2
        return mean, np.maximum(variance, 1e-12)
