"""Benchmark runner for comparing uncertainty-aware regression methods.

This module is the central orchestration point for the project. It loads data, creates
subsets of different sizes, trains multiple models, evaluates them under the same metrics,
then returns or saves the resulting benchmark table.

The idea is to keep the research protocol consistent across all methods: same datasets,
standardized features, matched train/test splits, and identical statistical reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from data_preparation import prepare_dataset_for_benchmark, subsample_training_data
from metrics import compute_regression_metrics


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run.

    This class defines the grid over datasets, training sizes, repetitions, and model
    functions used in the experimental loop.
    """

    dataset_names: Iterable[str]
    train_sizes: Iterable[int]
    n_splits: int = 20
    random_seed: int = 0
    test_fraction: float = 0.2


def run_single_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_factory: Callable[..., Any],
    model_name: str,
    **kwargs,
) -> Dict[str, Any]:
    """Train one model and compute its metrics on a single split.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test targets.
    model_factory : Callable
        Function that builds and trains the model.
    model_name : str
        Human-readable model label.
    **kwargs
        Optional arguments passed to the model factory.

    Returns
    -------
    dict
        Dictionary containing the model name, metadata, and metrics.
    """
    model = model_factory(X_train, y_train, **kwargs)
    mean, std = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, mean, std)

    return {
        "model": model_name,
        "rmse": metrics["rmse"],
        "nll": metrics["nll"],
        "crps": metrics["crps"],
        "coverage_90": metrics["coverage_90"],
    }


def benchmark_models(
    model_factories: Dict[str, Callable[..., Any]],
    dataset_names: Iterable[str],
    train_sizes: Iterable[int],
    n_splits: int = 20,
    dataset_loader: Optional[Callable[[str], Dict[str, np.ndarray]]] = None,
) -> pd.DataFrame:
    """Run the benchmark across datasets, training sizes, and model families.

    This is the main experimental loop. It performs the repeated train/test protocol and
    builds a tabular result set that can later be exported to CSV or plotted as learning
    curves. A custom ``dataset_loader`` can be supplied for synthetic or custom datasets.
    """
    rows: List[dict] = []
    loader = dataset_loader or prepare_dataset_for_benchmark

    for dataset_name in dataset_names:
        for split_idx in range(n_splits):
            data = loader(dataset_name, random_seed=split_idx)
            X_train, X_test, y_train, y_test = (
                data["X_train"],
                data["X_test"],
                data["y_train"],
                data["y_test"],
            )

            for train_size in train_sizes:
                X_sub, y_sub = subsample_training_data(X_train, y_train, train_size, random_seed=split_idx)

                for model_name, model_factory in model_factories.items():
                    result = run_single_experiment(
                        X_sub,
                        y_sub,
                        X_test,
                        y_test,
                        model_factory,
                        model_name,
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "split": split_idx,
                            "train_size": train_size,
                            **result,
                        }
                    )

    return pd.DataFrame(rows)


__all__ = [
    "BenchmarkConfig",
    "run_single_experiment",
    "benchmark_models",
]
