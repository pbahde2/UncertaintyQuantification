"""Exact Gaussian process baseline implemented with BoTorch."""

from __future__ import annotations

import copy
from functools import partial

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from ._common import (
    TORCH_DTYPE,
    TargetStandardizer,
    TrainingObjectiveStoppingCriterion,
    as_2d,
    set_seed,
)


class ExactGPModel:
    """Matérn-5/2 ARD GP with learned output scale and observation noise."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float = 0.01,
        max_iterations: int = 400,
        warmup_iterations: int = 50,
        patience: int = 5,
        min_delta: float = 1e-4,
        random_seed: int = 0,
    ) -> None:
        if (
            learning_rate <= 0
            or max_iterations <= 0
            or warmup_iterations <= 0
            or patience <= 0
            or min_delta < 0
        ):
            raise ValueError(
                "learning_rate, max_iterations, warmup_iterations, and patience must "
                "be positive and min_delta must be non-negative"
            )
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.warmup_iterations = warmup_iterations
        self.patience = patience
        self.min_delta = min_delta
        self.random_seed = random_seed
        self.fit(X_train, y_train)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ExactGPModel":
        set_seed(self.random_seed)
        self.target_scaler = TargetStandardizer.fit(y_train)
        train_x = torch.as_tensor(as_2d(X_train), dtype=TORCH_DTYPE)
        train_y = torch.as_tensor(
            self.target_scaler.transform(y_train), dtype=TORCH_DTYPE
        ).unsqueeze(-1)
        # Define kernel.
        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]))
        likelihood = GaussianLikelihood()
        likelihood.noise = 0.1
        # Define model.
        self.model = SingleTaskGP(
            train_x, train_y, likelihood=likelihood,
            covar_module=kernel, mean_module=ConstantMean()
        )
        self.likelihood = self.model.likelihood
        # Fit through BoTorch's dispatcher while explicitly using its torch path.
        self.model.train()
        self.likelihood.train()
        objective = ExactMarginalLogLikelihood(self.likelihood, self.model)
        stopping_criterion = TrainingObjectiveStoppingCriterion(
            first_monitored_step=min(
                self.warmup_iterations, self.max_iterations
            ),
            patience=self.patience,
            min_delta=self.min_delta,
        )
        best_state = copy.deepcopy(self.model.state_dict())

        def retain_best_state(parameters, result) -> None:
            del parameters, result
            nonlocal best_state
            if stopping_criterion.improved:
                best_state = copy.deepcopy(self.model.state_dict())

        fit_gpytorch_mll(
            objective,
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={
                "step_limit": self.max_iterations,
                "optimizer": partial(torch.optim.Adam, lr=self.learning_rate),
                "stopping_criterion": stopping_criterion,
                "callback": retain_best_state,
            },
            max_attempts=1,
        )
        self.model.load_state_dict(best_state)
        self.best_epoch_ = stopping_criterion.best_step
        self.training_objective_ = stopping_criterion.best_loss
        self.optimization_steps_ = stopping_criterion.step
        # Exact-GP optimization is full-batch: one step is one epoch.
        self.optimization_epochs_ = stopping_criterion.step
        self.optimization_status_ = (
            "converged" if stopping_criterion.converged else "max_iterations"
        )
        self.model.eval()
        self.likelihood.eval()
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
