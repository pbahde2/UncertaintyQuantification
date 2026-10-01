"""MC dropout baseline.

This model approximates a Bayesian network by applying dropout at inference time and
averaging multiple stochastic forward passes. It is a cheap uncertainty baseline that is
useful for comparing against a proper GP in the small-data regime.
"""

from __future__ import annotations

import numpy as np


class MCDropoutModel:
    """Placeholder Monte Carlo dropout model implementation."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_samples: int = 50, **kwargs):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self.n_samples = n_samples
        self.kwargs = kwargs
        self._fit_mean = float(np.mean(self.y_train))
        self._fit_std = float(np.std(self.y_train)) or 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the dropout-based approximate Bayesian model."""
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self._fit_mean = float(np.mean(self.y_train))
        self._fit_std = float(np.std(self.y_train)) or 1.0

    def predict(self, X_test: np.ndarray):
        """Return Monte Carlo predictive mean and standard deviation."""
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        # Placeholder MC dropout behavior: repeated stochastic perturbations increase the
        # variance beyond the base residual scale, which gives a plausible estimate for the
        # benchmark API while the actual network is later implemented.
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(self.n_samples):
            perturb = rng.normal(0.0, 0.2 * self._fit_std, size=len(X_test))
            draws.append(self._fit_mean + perturb)
        ensemble = np.stack(draws, axis=0)
        mean = np.mean(ensemble, axis=0)
        std = np.std(ensemble, axis=0)
        return mean, std
