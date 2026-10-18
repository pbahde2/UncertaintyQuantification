"""Sparse variational Gaussian process baseline."""

from __future__ import annotations

import copy
import numpy as np
import torch
from botorch.models import SingleTaskVariationalGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO

from ._common import TORCH_DTYPE, TargetStandardizer, as_2d, gaussian_nll, set_seed


class SparseGPModel:
    """SVGP with learned inducing locations and a Gaussian likelihood."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_inducing: int = 64,
                 learning_rate: float = 0.01, max_epochs: int = 400,
                 patience: int = 5, warmup_epochs: int = 50,
                 min_delta: float = 1e-4, random_seed: int = 0,
                 X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                 **_: object):
        self.n_inducing = n_inducing
        self.learning_rate, self.max_epochs = learning_rate, max_epochs
        self.patience, self.warmup_epochs = patience, warmup_epochs
        self.min_delta, self.random_seed = min_delta, random_seed
        self.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "SparseGPModel":
        set_seed(self.random_seed)
        self.target_scaler = TargetStandardizer.fit(y_train)
        train_x = torch.as_tensor(as_2d(X_train), dtype=TORCH_DTYPE)
        train_y = torch.as_tensor(
            self.target_scaler.transform(y_train), dtype=TORCH_DTYPE
        ).unsqueeze(-1)
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be provided or both be omitted")
        val_x = (
            torch.as_tensor(as_2d(X_val), dtype=TORCH_DTYPE)
            if X_val is not None and len(X_val) else None
        )
        val_y = (
            torch.as_tensor(self.target_scaler.transform(y_val), dtype=TORCH_DTYPE)
            if val_x is not None else None
        )
        # Cap the requested configuration when fewer training observations are available.
        count = min(self.n_inducing, len(train_x))
        self.n_inducing_ = count
        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]))
        likelihood = GaussianLikelihood()
        likelihood.noise = 0.1
        self.model = SingleTaskVariationalGP(
            train_x, train_y, inducing_points=count, learn_inducing_points=True,
            likelihood=likelihood, covar_module=kernel, mean_module=ConstantMean(),
        )
        self.likelihood = self.model.likelihood
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        objective = VariationalELBO(self.likelihood, self.model.model, num_data=len(train_y))
        best_loss, best_epoch = float("inf"), 0
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        # A variational GP can initially make validation NLL worse while its inducing
        # distribution moves away from the prior. Do not let that transient select an
        # effectively untrained one-epoch model.
        first_monitored_epoch = min(max(1, self.warmup_epochs), self.max_epochs)
        for epoch in range(1, self.max_epochs + 1):
            optimizer.zero_grad()
            loss = -objective(self.model.model(train_x), train_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            if val_x is not None and epoch >= first_monitored_epoch:
                self.model.eval()
                self.likelihood.eval()
                with torch.no_grad():
                    predictive = self.likelihood(self.model.model(val_x))
                    selection_loss = float(
                        gaussian_nll(predictive.mean, predictive.variance, val_y)
                    )
                self.model.train()
                self.likelihood.train()
                # Keep the epoch only when validation NLL improves by at least min_delta.
                if selection_loss < best_loss - self.min_delta:
                    best_loss = selection_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    stale = 0
                else:
                    stale += 1
                    # Stop after the configured number of post-warm-up non-improvements.
                    if stale >= self.patience:
                        break
        if val_x is not None:
            # Bypass BoTorch's exact-GP-specific loader; this wrapper has no train_targets.
            torch.nn.Module.load_state_dict(self.model, best_state)
            self.best_epoch_ = best_epoch
        else:
            self.best_epoch_ = self.max_epochs
        self.model.eval()
        self.likelihood.eval()
        return self

    def refit_full(self, X_train: np.ndarray, y_train: np.ndarray) -> "SparseGPModel":
        """Reinitialize and train on all labels for the selected duration."""
        # Preserve both validation-selected duration and model capacity. In particular,
        # do not increase the inducing-point count merely because validation observations
        # rejoin the final training set.
        selected_epochs = self.best_epoch_
        effective_inducing = self.n_inducing_
        maximum, requested_inducing = self.max_epochs, self.n_inducing
        self.max_epochs = selected_epochs
        self.n_inducing = effective_inducing
        try:
            # With no validation arguments, fit uses every observation for exactly this many epochs.
            self.fit(X_train, y_train)
        finally:
            self.max_epochs = maximum
            self.n_inducing = requested_inducing
        self.best_epoch_ = selected_epochs
        return self

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        test_x = torch.as_tensor(as_2d(X_test), dtype=TORCH_DTYPE)
        with torch.no_grad():
            predictive = self.model.posterior(test_x, observation_noise=True)
        mean = (
            predictive.mean.squeeze(-1).detach().numpy() * self.target_scaler.std
            + self.target_scaler.mean
        )
        variance = predictive.variance.squeeze(-1).detach().numpy() * self.target_scaler.std**2
        return mean, np.maximum(variance, 1e-12)
