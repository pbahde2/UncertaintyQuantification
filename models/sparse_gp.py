"""Sparse variational GP baseline.

This model approximates the exact GP using inducing points. It provides a cheaper GP
baseline for the benchmark and is important for studying how approximation quality changes
as the data set size decreases.
"""

from __future__ import annotations

import numpy as np


class SparseGPModel:
    """Sparse variational GP model with a simple placeholder posterior."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_inducing: int = 64, **kwargs):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self.n_inducing = n_inducing
        self.kwargs = kwargs
        self._fit_mean = float(np.mean(self.y_train))
        self._fit_std = float(np.std(self.y_train)) or 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the sparse GP model to the supplied training data."""
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self._fit_mean = float(np.mean(self.y_train))
        self._fit_std = float(np.std(self.y_train)) or 1.0

    def predict(self, X_test: np.ndarray):
        """Return an approximate predictive mean and standard deviation."""
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        # Placeholder behavior: a robust yet smooth pseudo-posterior based on local distance.
        # This keeps the benchmark interface consistent while the actual GPyTorch variational
        # implementation is added later.
        means = np.full(len(X_test), self._fit_mean, dtype=float)
        if len(self.X_train) == 0:
            std = np.full(len(X_test), self._fit_std, dtype=float)
            return means, std

        diffs = self.X_train[:, None, :] - X_test[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        nearest = np.min(dists, axis=0)
        std = self._fit_std * (1.0 / (1.0 + nearest))
        return means, std
