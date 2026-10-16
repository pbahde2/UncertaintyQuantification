"""Utilities for loading datasets, preprocessing features, and creating train/test subsets.

This module keeps all data-preparation logic in one place so that every model is trained
and evaluated under the same protocol. The code is intentionally lightweight and designed
for a benchmark study comparing Gaussian processes, deep ensembles, MC dropout, and DKL.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_uci_regression_dataset(name: str, version: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load a UCI regression dataset.

    Parameters
    ----------
    name : str
        Dataset name, typically the OpenML identifier or a known dataset key.
    version : str | None, optional
        Optional version string passed through for dataset retrieval.

    Returns
    -------
    X : np.ndarray
        Feature matrix with shape (n_samples, n_features).
    y : np.ndarray
        Regression target with shape (n_samples,).

    Notes
    -----
    This project is meant to evaluate the small-data regime, so all datasets should be
    standardized before model training. The current implementation is intentionally simple
    and can be replaced by dataset-specific loaders if needed.
    """
    # This is a lightweight placeholder for the benchmark pipeline.
    # Real-world runs may use OpenML or a local CSV store instead.
    dataset = fetch_openml(name=name, version=version, as_frame=False, parser="auto")
    X, y = dataset.data, dataset.target
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def standardize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a standard scaler on the training data and transform both splits.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    X_test : np.ndarray
        Test features.

    Returns
    -------
    X_train_scaled : np.ndarray
        Training features after scaling.
    X_test_scaled : np.ndarray
        Test features after scaling.
    scaler : StandardScaler
        Fitted scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def create_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.8,
    random_seed: int = 0,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a train/test split using the sklearn convention.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix.
    y : np.ndarray
        Full target vector.
    train_size : float, optional
        Fraction of data used for training. Default is 0.8.
    random_seed : int, optional
        Random seed controlling split reproducibility.
    shuffle : bool, optional
        Whether to shuffle the data before splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of np.ndarray
        Split arrays.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_seed,
        shuffle=shuffle,
    )
    return X_train, X_test, y_train, y_test


def subsample_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_points: int,
    random_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a smaller training subset of size ``n_points``.

    This function is used to sweep the training set size across the small-data regime,
    such as n in {50, 100, 200, 500, 1000}. It keeps the subset selection deterministic
    for a given seed and allows a fair comparison across methods.
    """
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(X_train), size=min(n_points, len(X_train)), replace=False)
    return X_train[idx], y_train[idx]


def nested_training_subsets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_sizes: Iterable[int],
    random_seed: int = 0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Create reproducible nested subsets from one random ordering.

    Every smaller requested subset is a prefix of every larger requested subset.
    This reduces learning-curve noise caused by independently resampling each size.
    """
    sizes = [int(size) for size in train_sizes]
    if any(size <= 0 for size in sizes):
        raise ValueError("all training sizes must be positive")
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    if len(X) != len(y):
        raise ValueError("X_train and y_train must contain the same number of observations")
    order = np.random.default_rng(random_seed).permutation(len(X))
    return {
        size: (X[order[: min(size, len(X))]], y[order[: min(size, len(y))]])
        for size in sizes
    }


def split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split an ordered subset while preserving nesting across subset sizes.

    Validation positions are selected by a deterministic fractional schedule. Therefore,
    extending a prefix never changes the role of an observation already present.
    """
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of observations")
    positions = np.arange(len(X))
    validation_mask = (
        np.floor((positions + 1) * validation_fraction)
        > np.floor(positions * validation_fraction)
    )
    if np.all(validation_mask):
        validation_mask[-1] = False
    return X[~validation_mask], X[validation_mask], y[~validation_mask], y[validation_mask]


def prepare_dataset_for_benchmark(
    dataset_name: str,
    train_fraction: float = 0.8,
    train_size: int | None = None,
    random_seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Load a dataset and prepare a reproducible train/test split.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.
    train_fraction : float, optional
        Fraction of data used for training. Default is 0.8.
    train_size : int | None, optional
        If provided, it overrides the fraction and selects a fixed number of points.
    random_seed : int, optional
        Random seed used for all stochastic steps.

    Returns
    -------
    dict
        Dictionary containing the raw split arrays:
        {'X_train': ..., 'X_test': ..., 'y_train': ..., 'y_test': ...}
    """
    X, y = load_uci_regression_dataset(dataset_name)

    if train_size is not None:
        X_train, X_test, y_train, y_test = create_train_test_split(
            X,
            y,
            train_size=train_size / len(X),
            random_seed=random_seed,
        )
    else:
        X_train, X_test, y_train, y_test = create_train_test_split(
            X,
            y,
            train_size=train_fraction,
            random_seed=random_seed,
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


__all__ = [
    "load_uci_regression_dataset",
    "standardize_features",
    "create_train_test_split",
    "subsample_training_data",
    "nested_training_subsets",
    "split_train_validation",
    "prepare_dataset_for_benchmark",
]
