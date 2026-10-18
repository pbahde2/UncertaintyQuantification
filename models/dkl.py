"""Deep kernel learning with a jointly trained neural feature map and exact GP."""

from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
import gpytorch

from ._common import TORCH_DTYPE, TargetStandardizer, as_2d, gaussian_nll, set_seed


class _FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        width = max(2, 2 * input_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, width, dtype=TORCH_DTYPE), nn.ReLU(),
            nn.Linear(width, width, dtype=TORCH_DTYPE), nn.ReLU(),
            # Preserve the input dimensionality while learning a nonlinear warping.
            nn.Linear(width, input_dim, dtype=TORCH_DTYPE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _DKLGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        feature_dim = train_x.shape[-1]
        self.feature_extractor = _FeatureExtractor(feature_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feature_dim)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        features = self.feature_extractor(x)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(features), self.covar_module(features)
        )


class DKLModel:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 learning_rate: float = 0.01, max_epochs: int = 400,
                 patience: int = 5, warmup_epochs: int = 50,
                 min_delta: float = 1e-4, random_seed: int = 0,
                 X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                 **_: object):
        self.learning_rate, self.max_epochs = learning_rate, max_epochs
        self.patience, self.warmup_epochs = patience, warmup_epochs
        self.min_delta, self.random_seed = min_delta, random_seed
        self.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "DKLModel":
        set_seed(self.random_seed)
        self.target_scaler = TargetStandardizer.fit(y_train)
        train_x = torch.as_tensor(as_2d(X_train), dtype=TORCH_DTYPE)
        train_y = torch.as_tensor(
            self.target_scaler.transform(y_train), dtype=TORCH_DTYPE
        )
        self.feature_dim = train_x.shape[-1]
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
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = 0.1
        self.model = _DKLGP(train_x, train_y, self.likelihood).to(dtype=TORCH_DTYPE)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        objective = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        best_loss, best_epoch = float("inf"), 0
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        first_monitored_epoch = min(max(1, self.warmup_epochs), self.max_epochs)
        for epoch in range(1, self.max_epochs + 1):
            optimizer.zero_grad()
            loss = -objective(self.model(train_x), train_y)
            loss.backward()
            optimizer.step()
            if val_x is not None and epoch >= first_monitored_epoch:
                self.model.eval()
                self.likelihood.eval()
                with torch.no_grad():
                    predictive = self.likelihood(self.model(val_x))
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
                    # Five consecutive non-improving epochs trigger stopping by default.
                    if stale >= self.patience:
                        break
        if val_x is not None:
            self.model.load_state_dict(best_state)
            self.best_epoch_ = best_epoch
        else:
            self.best_epoch_ = self.max_epochs
        self.model.eval()
        self.likelihood.eval()
        return self

    def refit_full(self, X_train: np.ndarray, y_train: np.ndarray) -> "DKLModel":
        """Reinitialize and train on all labels for the selected duration."""
        # Preserve the validation-selected duration before fit creates a fresh DKL model.
        selected_epochs, maximum = self.best_epoch_, self.max_epochs
        self.max_epochs = selected_epochs
        # With no validation arguments, fit uses every observation for exactly this many epochs.
        self.fit(X_train, y_train)
        self.max_epochs = maximum
        self.best_epoch_ = selected_epochs
        return self

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        test_x = torch.as_tensor(as_2d(X_test), dtype=TORCH_DTYPE)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive = self.likelihood(self.model(test_x))
        mean = predictive.mean.numpy() * self.target_scaler.std + self.target_scaler.mean
        variance = predictive.variance.numpy() * self.target_scaler.std**2
        return mean, np.maximum(variance, 1e-12)
