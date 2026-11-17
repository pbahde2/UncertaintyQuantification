"""Sparse variational Gaussian process baseline."""

from __future__ import annotations

import copy

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll_torch
from botorch.models import SingleTaskVariationalGP
from botorch.optim.closures import get_loss_closure_with_grads
from botorch.optim.utils import get_data_loader
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO

from ._common import (
    TORCH_DTYPE,
    TargetStandardizer,
    as_2d,
    set_seed,
)


class EpochELBOStoppingCriterion:
    """Stop after complete epochs without sufficient mean-ELBO improvement."""

    def __init__(
        self,
        batches_per_epoch: int,
        first_monitored_epoch: int = 50,
        patience: int = 5,
        min_delta: float = 1e-4,
    ) -> None:
        if batches_per_epoch <= 0 or first_monitored_epoch <= 0 or patience <= 0:
            raise ValueError(
                "batches_per_epoch, first_monitored_epoch, and patience must be positive"
            )
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.batches_per_epoch = batches_per_epoch
        self.first_monitored_epoch = first_monitored_epoch
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def reset(self) -> None:
        self.batch_losses: list[torch.Tensor] = []
        self.epoch_losses: list[float] = []
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.stale_epochs = 0
        self.improved = False
        self.converged = False

    def __call__(self, loss: torch.Tensor) -> bool:
        self.improved = False
        self.batch_losses.append(loss.detach())
        if len(self.batch_losses) < self.batches_per_epoch:
            return False

        epoch_loss = float(torch.stack(self.batch_losses).mean())
        self.batch_losses.clear()
        self.epoch_losses.append(epoch_loss)
        epoch = len(self.epoch_losses)
        if epoch < self.first_monitored_epoch:
            return False

        if epoch_loss < self.best_loss - self.min_delta:
            self.best_loss = epoch_loss
            self.best_epoch = epoch
            self.stale_epochs = 0
            self.improved = True
        else:
            self.stale_epochs += 1

        self.converged = self.stale_epochs >= self.patience
        return self.converged


class SparseGPModel:
    """SVGP with learned inducing locations and a Gaussian likelihood."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_inducing: int = 64,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        max_epochs: int = 400,
        warmup_epochs: int = 50,
        patience: int = 5,
        min_delta: float = 1e-4,
        random_seed: int = 0,
    ) -> None:
        self.n_inducing = n_inducing
        if (
            batch_size <= 0
            or learning_rate <= 0
            or max_epochs <= 0
            or warmup_epochs <= 0
            or patience <= 0
        ):
            raise ValueError(
                "batch_size, learning_rate, max_epochs, warmup_epochs, and patience "
                "must be positive"
            )
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.random_seed = random_seed
        self.fit(X_train, y_train)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SparseGPModel":
        set_seed(self.random_seed)
        self.target_scaler = TargetStandardizer.fit(y_train)
        train_x = torch.as_tensor(as_2d(X_train), dtype=TORCH_DTYPE)
        train_y = torch.as_tensor(
            self.target_scaler.transform(y_train), dtype=TORCH_DTYPE
        ).unsqueeze(-1)
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
        objective = VariationalELBO(
            self.likelihood, self.model.model, num_data=len(train_y)
        )
        data_loader = get_data_loader(
            self.model.model,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        self.batch_size_ = data_loader.batch_size
        parameters = {
            name: parameter
            for name, parameter in objective.named_parameters()
            if parameter.requires_grad
        }
        closure = get_loss_closure_with_grads(
            objective,
            parameters=parameters,
            data_loader=data_loader,
        )
        stopping_criterion = EpochELBOStoppingCriterion(
            batches_per_epoch=len(data_loader),
            first_monitored_epoch=min(self.warmup_epochs, self.max_epochs),
            patience=self.patience,
            min_delta=self.min_delta,
        )
        best_state = copy.deepcopy(self.model.state_dict())

        def retain_best_state(parameters, result) -> None:
            del parameters, result
            nonlocal best_state
            if stopping_criterion.improved:
                best_state = copy.deepcopy(self.model.state_dict())

        result = fit_gpytorch_mll_torch(
            objective,
            parameters=parameters,
            closure=closure,
            step_limit=self.max_epochs * len(data_loader),
            stopping_criterion=stopping_criterion,
            optimizer=lambda params: torch.optim.Adam(
                params, lr=self.learning_rate
            ),
            callback=retain_best_state,
        )
        # Bypass BoTorch's exact-GP-specific loader: the variational wrapper
        # does not expose ``train_targets``.
        torch.nn.Module.load_state_dict(self.model, best_state)
        self.best_epoch_ = stopping_criterion.best_epoch
        self.best_elbo_epoch_ = stopping_criterion.best_epoch
        self.optimization_epochs_ = len(stopping_criterion.epoch_losses)
        self.optimization_steps_ = result.step
        self.best_training_elbo_ = -stopping_criterion.best_loss
        self.optimization_status_ = (
            "converged" if stopping_criterion.converged else "max_epochs"
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
