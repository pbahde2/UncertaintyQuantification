"""Synthetic benchmark entry point for a first executable GP study.

This script performs a minimal but realistic small-data benchmark on a synthetic regression
problem and reports the scores for the exact GP baseline. It is intentionally kept simple,
so it serves as a working proof-of-concept before the full UCI sweep is added.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execute_experiment import ModelSpec, execute_experiment
from models.deep_ensemble import DeepEnsembleModel
from models.dkl import DKLModel
from models.exact_gp import ExactGPModel
from models.mc_dropout import MCDropoutModel
from models.sparse_gp import SparseGPModel


def synthetic_regression_dataset(dataset_name: str = "synthetic", random_seed: int = 0):
    """Build a synthetic 1D regression dataset with a nonlinear target function.

    The task is intentionally simple, but it captures the type of problem used to examine
    uncertainty behavior in the small-data regime.
    """
    rng = np.random.default_rng(random_seed)
    X = rng.uniform(-3.0, 3.0, size=(300, 1))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 0] + 0.2 * rng.normal(size=len(X))

    return {
        "X_train": X[:240],
        "X_test": X[240:],
        "y_train": y[:240],
        "y_test": y[240:],
    }


def poc_experiment() -> None:
    """Run a compact benchmark comparing all five model families."""
    model_factories = {
        "exact_gp": lambda X_train, y_train, **kwargs: ExactGPModel(X_train, y_train, **kwargs),
        "sparse_gp_m16": ModelSpec(SparseGPModel, n_inducing=16),
        "sparse_gp_m64": ModelSpec(SparseGPModel, n_inducing=64),
        "sparse_gp_m256": ModelSpec(SparseGPModel, n_inducing=256),
        "deep_ensemble": lambda X_train, y_train, **kwargs: DeepEnsembleModel(
            X_train, y_train, n_members=5, **kwargs
        ),
        "mc_dropout": lambda X_train, y_train, **kwargs: MCDropoutModel(
            X_train, y_train, n_samples=25, **kwargs
        ),
        "dkl": lambda X_train, y_train, **kwargs: DKLModel(X_train, y_train, **kwargs),
    }

    result_df = execute_experiment(
        model_factories=model_factories,
        dataset_names=["synthetic"],
        train_sizes=[25, 50, 100, 150],
        n_splits=3,
        dataset_loader=synthetic_regression_dataset,
        verbose=True,
    )

    output_path = ROOT / "results" / "tables" / "poc_experiment.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    summary = (
        result_df.groupby(["model", "train_size"], as_index=False)
        [["rmse", "nll", "crps", "coverage_90"]]
        .mean()
        .sort_values(["model", "train_size"])
    )

    print(summary)
    print(f"\nSaved benchmark table to: {output_path}")
    print("\nAll eligible model configurations were benchmarked successfully.")


if __name__ == "__main__":
    poc_experiment()
