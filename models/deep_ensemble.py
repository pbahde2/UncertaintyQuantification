"""Deep ensemble with heteroscedastic Gaussian member networks."""

from __future__ import annotations

import copy
import numpy as np
import torch

from ._common import (
    DEFAULT_LEARNING_RATE,
    TORCH_DTYPE,
    GaussianMLP,
    TargetStandardizer,
    as_2d,
    gaussian_nll,
    set_seed,
)


class DeepEnsembleModel:
    """Five independently initialized MLP Gaussian regressors by default."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 n_members: int = 5,
                 hidden_size: int | None = None,
                 learning_rate: float = DEFAULT_LEARNING_RATE, max_epochs: int = 400,
                 batch_size: int = 32, patience: int = 5,
                 warmup_epochs: int = 50, min_delta: float = 1e-4,
                 random_seed: int = 0):
        self.n_members = n_members
        self.hidden_size = hidden_size
        self.learning_rate, self.max_epochs = learning_rate, max_epochs
        self.batch_size, self.patience = batch_size, patience
        self.warmup_epochs = warmup_epochs
        self.min_delta = min_delta
        self.random_seed = random_seed
        self.members: list[GaussianMLP] = []
        self.fit(X_train, y_train)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DeepEnsembleModel":
        X = as_2d(X_train)
        y_original = np.asarray(y_train).reshape(-1)
        if len(X) != len(y_original):
            raise ValueError("X_train and y_train must have equal length")
        if (
            self.n_members <= 0
            or self.max_epochs <= 0
            or self.batch_size <= 0
            or self.patience <= 0
        ):
            raise ValueError(
                "n_members, max_epochs, batch_size, and patience must be positive"
            )
        self.target_scaler = TargetStandardizer.fit(y_original)
        y = self.target_scaler.transform(y_original)
        X_tensor = torch.as_tensor(X, dtype=TORCH_DTYPE)
        y_tensor = torch.as_tensor(y, dtype=TORCH_DTYPE)
        indices = torch.arange(len(X_tensor))

        self.members = []
        self.selected_epochs_: list[int] = []
        self.training_epochs_: list[int] = []
        self.training_nlls_: list[float] = []
        for member_idx in range(self.n_members):
            seed = self.random_seed + member_idx
            set_seed(seed)
            member = GaussianMLP(X.shape[1], hidden_size=self.hidden_size)
            optimizer = torch.optim.Adam(
                member.parameters(), lr=self.learning_rate
            )
            generator = torch.Generator().manual_seed(seed)
            best_loss = float("inf")
            best_state = copy.deepcopy(member.state_dict())
            best_epoch = 0
            stale_epochs = 0
            first_monitored_epoch = min(
                max(1, self.warmup_epochs), self.max_epochs
            )

            for epoch in range(1, self.max_epochs + 1):
                member.train()
                shuffled = indices[torch.randperm(len(indices), generator=generator)]
                for start in range(0, len(shuffled), self.batch_size):
                    batch = shuffled[start : start + self.batch_size]
                    optimizer.zero_grad()
                    mean, variance = member(X_tensor[batch])
                    loss = gaussian_nll(mean, variance, y_tensor[batch])
                    loss.backward()
                    optimizer.step()

                if epoch < first_monitored_epoch:
                    continue

                # As in the original implementation, each member selects and
                # restores its own checkpoint. The monitored Gaussian NLL is
                # now evaluated on all training observations instead of a
                # separate validation set.
                member.eval()
                with torch.no_grad():
                    mean, variance = member(X_tensor)
                    training_loss = float(
                        gaussian_nll(mean, variance, y_tensor)
                    )
                if training_loss < best_loss - self.min_delta:
                    best_loss = training_loss
                    best_state = copy.deepcopy(member.state_dict())
                    best_epoch = epoch
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.patience:
                        break

            member.load_state_dict(best_state)
            member.eval()
            self.members.append(member)
            self.selected_epochs_.append(best_epoch)
            self.training_epochs_.append(epoch)
            self.training_nlls_.append(best_loss)
        self.optimization_epochs_ = float(np.mean(self.training_epochs_))
        self.training_nll_ = float(np.mean(self.training_nlls_))
        return self

    def predict_components(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = torch.as_tensor(as_2d(X_test), dtype=TORCH_DTYPE)
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
