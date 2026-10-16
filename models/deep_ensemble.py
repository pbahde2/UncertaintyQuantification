"""Deep ensemble with heteroscedastic Gaussian member networks."""

from __future__ import annotations

import numpy as np
import torch

from ._common import GaussianMLP, TargetStandardizer, as_2d, set_seed, train_gaussian_network


class DeepEnsembleModel:
    """Five independently initialized MLP Gaussian regressors by default."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_members: int = 5,
                 learning_rate: float = 1e-3, max_epochs: int = 400,
                 batch_size: int = 32, patience: int = 5, warmup_epochs: int = 50,
                 min_delta: float = 1e-4,
                 validation_fraction: float = 0.2, random_seed: int = 0,
                 X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                 **_: object):
        self.n_members = n_members
        self.learning_rate, self.max_epochs = learning_rate, max_epochs
        self.batch_size, self.patience = batch_size, patience
        self.warmup_epochs, self.min_delta = warmup_epochs, min_delta
        self.validation_fraction, self.random_seed = validation_fraction, random_seed
        self.members: list[GaussianMLP] = []
        self.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "DeepEnsembleModel":
        X = as_2d(X_train)
        self.target_scaler = TargetStandardizer.fit(y_train)
        y = self.target_scaler.transform(y_train)
        y_val_scaled = self.target_scaler.transform(y_val) if y_val is not None else None
        self.members = []
        self.selected_epochs_: list[int] = []
        for member_idx in range(self.n_members):
            seed = self.random_seed + member_idx
            set_seed(seed)
            member = GaussianMLP(X.shape[1])
            best_epoch = train_gaussian_network(
                member, X, y, learning_rate=self.learning_rate, max_epochs=self.max_epochs,
                batch_size=self.batch_size, patience=self.patience,
                warmup_epochs=self.warmup_epochs, min_delta=self.min_delta,
                validation_fraction=self.validation_fraction, seed=seed,
                X_val=X_val, y_val=y_val_scaled,
            )
            member.eval()
            self.members.append(member)
            self.selected_epochs_.append(best_epoch)
        return self

    def refit_full(self, X_train: np.ndarray, y_train: np.ndarray) -> "DeepEnsembleModel":
        """Reinitialize members and train on all labels for their selected durations."""
        # Each independently trained member keeps its own validation-selected duration.
        selected_epochs = self.selected_epochs_.copy()
        X = as_2d(X_train)
        self.target_scaler = TargetStandardizer.fit(y_train)
        y = self.target_scaler.transform(y_train)
        self.members = []
        for member_idx, epochs in enumerate(selected_epochs):
            seed = self.random_seed + member_idx
            set_seed(seed)
            # Reinitialize this member instead of continuing from its selection checkpoint.
            member = GaussianMLP(X.shape[1])
            train_gaussian_network(
                member, X, y, learning_rate=self.learning_rate, max_epochs=epochs,
                batch_size=self.batch_size, patience=self.patience,
                warmup_epochs=self.warmup_epochs, min_delta=self.min_delta,
                validation_fraction=0.0, seed=seed, fixed_epochs=epochs,
            )
            member.eval()
            self.members.append(member)
        return self

    def predict_components(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = torch.as_tensor(as_2d(X_test), dtype=torch.float32)
        means, variances = [], []
        with torch.no_grad():
            for member in self.members:
                mean, variance = member(X)
                means.append(mean.numpy() * self.target_scaler.std + self.target_scaler.mean)
                variances.append(variance.numpy() * self.target_scaler.std**2)
        return np.stack(means), np.stack(variances)

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, variances = self.predict_components(X_test)
        mean = means.mean(axis=0)
        variance = (variances + means**2).mean(axis=0) - mean**2
        return mean, np.maximum(variance, 1e-12)
