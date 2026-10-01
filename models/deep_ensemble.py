"""Deep ensemble with a heteroscedastic Gaussian head.

This is the main neural baseline. Deep ensembles are the default UQ method in practice,
but they often assume much larger datasets than the small-data regime studied here.
"""

from __future__ import annotations

import numpy as np


class DeepEnsembleModel:
    """Placeholder deep ensemble model implementation."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_members: int = 5, **kwargs):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self.n_members = n_members
        self.kwargs = kwargs
        self._fit_mean = float(np.mean(self.y_train))
        self._fit_std = float(np.std(self.y_train)) or 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train an ensemble of neural networks with a Gaussian output head."""
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self._fit_mean = float(np.mean(self.y_train))
        self._fit_std = float(np.std(self.y_train)) or 1.0

    def predict(self, X_test: np.ndarray):
        """Return ensemble predictive mean and predictive standard deviation."""
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        # A lightweight placeholder: ensemble members differ slightly through random offsets,
        # mimicking the fact that ensemble predictions average mean estimates while the
        # predictive variance is increased by member disagreement.
        rng = np.random.default_rng(0)
        base_mean = np.full(len(X_test), self._fit_mean, dtype=float)
        member_noise = np.linspace(0.0, 0.25, self.n_members)
        ensemble_mean = np.mean(
            np.stack([base_mean + offset * self._fit_std * rng.normal(0.0, 1.0, size=len(X_test)) for offset in member_noise], axis=0),
            axis=0,
        )
        ensemble_std = np.full(len(X_test), self._fit_std * 0.5, dtype=float)
        return ensemble_mean, ensemble_std
