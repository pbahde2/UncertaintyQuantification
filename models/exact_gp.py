"""Exact Gaussian process baseline.

This model serves as the reference method in the small-data regime. The implementation is
kept intentionally lightweight but is mathematically a proper GP posterior using a Matérn-5/2-
like kernel in the exact inference sense. It provides a working baseline for the benchmark
before more advanced GPyTorch models are added.
"""

from __future__ import annotations

import numpy as np


class ExactGPModel:
    """Exact GP posterior with a Matérn-5/2 style kernel.

    The model fits the kernel hyperparameters from the training data in a simple data-driven
    way, then computes the predictive mean and variance using the closed-form GP equations.
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        self.X_train = self._as_2d(X_train)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self.lengthscale = kwargs.get("lengthscale", self._infer_lengthscale(self.X_train))
        self.noise = kwargs.get("noise", 1e-2)
        self.jitter = kwargs.get("jitter", 1e-6)

    def _as_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _infer_lengthscale(self, X: np.ndarray) -> float:
        """Roughly estimate a lengthscale from the median pairwise distance."""
        if len(X) < 2:
            return 1.0
        diffs = X[:, None, :] - X[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        upper = dists[np.triu_indices(len(X), k=1)]
        if upper.size == 0:
            return 1.0
        return float(np.median(upper[upper > 0])) + 1e-3

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute a Matérn-5/2 kernel matrix."""
        X1 = self._as_2d(X1)
        X2 = self._as_2d(X2)
        d = X1[:, None, :] - X2[None, :, :]
        r = np.linalg.norm(d, axis=2)
        coeff = np.sqrt(5.0) * r / self.lengthscale
        kernel = (1.0 + coeff + (5.0 * r**2) / (3.0 * self.lengthscale**2)) * np.exp(-coeff)
        return kernel

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the GP model on the supplied training data."""
        self.X_train = self._as_2d(X_train)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self.lengthscale = self._infer_lengthscale(self.X_train)

    def predict(self, X_test: np.ndarray):
        """Return the GP predictive mean and standard deviation."""
        X_test = self._as_2d(X_test)

        Kxx = self._kernel(self.X_train, self.X_train)
        Kxs = self._kernel(self.X_train, X_test)
        Kss = self._kernel(X_test, X_test)

        noise_matrix = self.noise**2 * np.eye(len(self.X_train))
        L = np.linalg.cholesky(Kxx + noise_matrix + self.jitter * np.eye(len(self.X_train)))

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        mean = Kxs.T @ alpha

        v = np.linalg.solve(L, Kxs)
        cov = Kss - v.T @ v
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))

        return mean.astype(float), std.astype(float)
