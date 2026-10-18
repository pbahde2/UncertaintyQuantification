"""Run a separate benchmark on UCI Power Plant and Electrical Grid Stability."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execute_experiment import ModelSpec, count_experiment_jobs, execute_experiment
from experiments.power_plant_experiment import (
    DATASET_NAME as POWER_PLANT_DATASET_NAME,
    N_FEATURES as POWER_PLANT_N_FEATURES,
    TEST_SIZE as POWER_PLANT_TEST_SIZE,
    TOTAL_OBSERVATIONS as POWER_PLANT_TOTAL_OBSERVATIONS,
    TRAIN_POOL_SIZE as POWER_PLANT_TRAIN_POOL_SIZE,
    UCI_DATA_ID as POWER_PLANT_UCI_DATA_ID,
    load_power_plant_data,
)
from models.deep_ensemble import DeepEnsembleModel
from models.dkl import DKLModel
from models.exact_gp import ExactGPModel
from models.mc_dropout import MCDropoutModel
from models.sparse_gp import SparseGPModel


GRID_DATASET_NAME = "electrical_grid_stability"
GRID_UCI_DATA_ID = 471
GRID_TOTAL_OBSERVATIONS = 10_000
# UCI supplies 12 candidate inputs. The documented non-predictive p1 column is removed.
GRID_N_FEATURES = 11
GRID_TRAIN_POOL_SIZE = 8_000
GRID_TEST_SIZE = 2_000
GRID_CACHE_PATH = ROOT / ".cache" / "uci" / "electrical_grid_stability_471.npz"

TRAIN_FRACTION = 0.8
TRAIN_SIZES = [50, 100, 200, 400, 800, 1_600]
N_SPLITS = 20
OUTPUT_PATH = ROOT / "results" / "tables" / "power_plant_grid_stability_experiment.csv"

DATASET_SPECS = {
    POWER_PLANT_DATASET_NAME: {
        "uci_data_id": POWER_PLANT_UCI_DATA_ID,
        "total_observations": POWER_PLANT_TOTAL_OBSERVATIONS,
        "n_features": POWER_PLANT_N_FEATURES,
        "train_pool_size": POWER_PLANT_TRAIN_POOL_SIZE,
        "test_size": POWER_PLANT_TEST_SIZE,
    },
    GRID_DATASET_NAME: {
        "uci_data_id": GRID_UCI_DATA_ID,
        "total_observations": GRID_TOTAL_OBSERVATIONS,
        "n_features": GRID_N_FEATURES,
        "train_pool_size": GRID_TRAIN_POOL_SIZE,
        "test_size": GRID_TEST_SIZE,
    },
}


def _extract_electrical_grid_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Select the continuous stability target and remove UCI's non-predictive p1 field."""
    features = dataset.data.features.drop(columns=["p1"], errors="raise")
    if "stab" not in dataset.data.targets:
        raise ValueError("UCI Electrical Grid data does not contain the 'stab' target")
    X = features.to_numpy(dtype=float)
    y = dataset.data.targets["stab"].to_numpy(dtype=float)
    return X, y


@lru_cache(maxsize=1)
def load_electrical_grid_data() -> tuple[np.ndarray, np.ndarray]:
    """Load, validate, and cache UCI dataset 471 for continuous regression."""
    if GRID_CACHE_PATH.exists():
        with np.load(GRID_CACHE_PATH) as cached:
            X = cached["X"]
            y = cached["y"]
    else:
        X, y = _extract_electrical_grid_arrays(fetch_ucirepo(id=GRID_UCI_DATA_ID))

    expected_X_shape = (GRID_TOTAL_OBSERVATIONS, GRID_N_FEATURES)
    expected_y_shape = (GRID_TOTAL_OBSERVATIONS,)
    if X.shape != expected_X_shape or y.shape != expected_y_shape:
        raise ValueError(
            "Unexpected Electrical Grid Stability shape: "
            f"X={X.shape}, y={y.shape}; expected {expected_X_shape} and "
            f"{expected_y_shape}."
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Electrical Grid Stability data contains non-finite values")

    if not GRID_CACHE_PATH.exists():
        GRID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GRID_CACHE_PATH, X=X, y=y)
    return X, y


def power_plant_grid_dataset(dataset_name: str, random_seed: int = 0):
    """Create one reproducible 80/20 partition for either benchmark dataset."""
    if dataset_name == POWER_PLANT_DATASET_NAME:
        X, y = load_power_plant_data()
    elif dataset_name == GRID_DATASET_NAME:
        X, y = load_electrical_grid_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=TRAIN_FRACTION,
        test_size=1.0 - TRAIN_FRACTION,
        random_state=random_seed,
        shuffle=True,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def build_model_factories():
    """Build the seven model configurations used by the shared protocol."""
    return {
        "exact_gp": lambda X_train, y_train, **kwargs: ExactGPModel(
            X_train, y_train, **kwargs
        ),
        "sparse_gp_m16": ModelSpec(SparseGPModel, n_inducing=16),
        "sparse_gp_m64": ModelSpec(SparseGPModel, n_inducing=64),
        "sparse_gp_m256": ModelSpec(SparseGPModel, n_inducing=256),
        "deep_ensemble": lambda X_train, y_train, **kwargs: DeepEnsembleModel(
            X_train, y_train, n_members=5, **kwargs
        ),
        "mc_dropout": lambda X_train, y_train, **kwargs: MCDropoutModel(
            X_train, y_train, n_samples=5, **kwargs
        ),
        "dkl": lambda X_train, y_train, **kwargs: DKLModel(
            X_train, y_train, **kwargs
        ),
    }


def power_plant_grid_stability_experiment() -> None:
    """Run the new benchmark without modifying the Power Plant/Protein result file."""
    model_factories = build_model_factories()
    dataset_names = list(DATASET_SPECS)
    dataset_lines = "\n".join(
        f"  {name}: {spec['total_observations']} rows, {spec['n_features']} features, "
        f"{spec['train_pool_size']} train-pool rows, {spec['test_size']} test rows"
        for name, spec in DATASET_SPECS.items()
    )
    print(
        f"Datasets:\n{dataset_lines}\n"
        f"Protocol: {N_SPLITS} splits per dataset\n"
        f"Training sizes: {TRAIN_SIZES}\n"
        f"Total model runs: "
        f"{count_experiment_jobs(model_factories, TRAIN_SIZES, len(dataset_names), N_SPLITS)}\n"
        f"Output: {OUTPUT_PATH}",
        flush=True,
    )

    result_df = execute_experiment(
        model_factories=model_factories,
        dataset_names=dataset_names,
        train_sizes=TRAIN_SIZES,
        n_splits=N_SPLITS,
        dataset_loader=power_plant_grid_dataset,
        verbose=True,
    )

    result_df.insert(
        3,
        "train_pool_size",
        result_df["dataset"].map(
            {name: spec["train_pool_size"] for name, spec in DATASET_SPECS.items()}
        ),
    )
    result_df.insert(
        4,
        "test_size",
        result_df["dataset"].map(
            {name: spec["test_size"] for name, spec in DATASET_SPECS.items()}
        ),
    )
    result_df.insert(
        5,
        "uci_data_id",
        result_df["dataset"].map(
            {name: spec["uci_data_id"] for name, spec in DATASET_SPECS.items()}
        ),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    summary = (
        result_df.groupby(["dataset", "model", "train_size"], as_index=False)[
            ["rmse", "nll", "crps", "coverage_90"]
        ]
        .mean()
        .sort_values(["dataset", "model", "train_size"])
    )
    print(summary)
    print(f"\nSaved benchmark table to: {OUTPUT_PATH}")


if __name__ == "__main__":
    power_plant_grid_stability_experiment()
