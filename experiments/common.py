"""Shared protocol configuration and model factories for both experiments."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from ucimlrepo import fetch_ucirepo

from execute_experiment import ModelSpec
from models.deep_ensemble import DeepEnsembleModel
from models.dkl import DKLModel
from models.exact_gp import ExactGPModel
from models.mc_dropout import MCDropoutModel
from models.sparse_gp import SparseGPModel


ROOT = Path(__file__).resolve().parents[1]

GRID_DATASET_NAME = "electrical_grid_stability"
GRID_UCI_DATA_ID = 471
GRID_CACHE_PATH = ROOT / ".cache" / "uci" / "electrical_grid_stability_471.npz"

POWER_PLANT_DATASET_NAME = "combined_cycle_power_plant"
POWER_PLANT_UCI_DATA_ID = 294
POWER_PLANT_CACHE_PATH = ROOT / ".cache" / "uci" / "power_plant_294.npz"

DATASET_NAMES = [POWER_PLANT_DATASET_NAME, GRID_DATASET_NAME]
UCI_DATA_IDS = {
    POWER_PLANT_DATASET_NAME: POWER_PLANT_UCI_DATA_ID,
    GRID_DATASET_NAME: GRID_UCI_DATA_ID,
}

VALIDATION_RATIO = 0.2
N_SPLITS = 20
MODEL_MAX_EPOCHS = 400
NEURAL_LEARNING_RATE = 0.001
GP_DKL_LEARNING_RATE = 0.01
DEEP_ENSEMBLE_MEMBERS = 5
MC_DROPOUT_SAMPLES = 5
SPARSE_GP_INDUCING_POINTS = (16, 64, 256)


def _validate_regression_data(
    X: np.ndarray, y: np.ndarray, dataset_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a loaded regression dataset without hard-coded dimensions."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"{dataset_name} features must be a two-dimensional array")
    if not len(X) or X.shape[1] == 0:
        raise ValueError(f"{dataset_name} must contain observations and features")
    if len(X) != len(y):
        raise ValueError(
            f"{dataset_name} has {len(X)} feature rows but {len(y)} targets"
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{dataset_name} contains non-finite values")
    return X, y


@lru_cache(maxsize=1)
def load_power_plant_data() -> tuple[np.ndarray, np.ndarray]:
    """Load, validate, and cache UCI Combined Cycle Power Plant data."""
    if POWER_PLANT_CACHE_PATH.exists():
        with np.load(POWER_PLANT_CACHE_PATH) as cached:
            X = cached["X"]
            y = cached["y"]
    else:
        dataset = fetch_ucirepo(id=POWER_PLANT_UCI_DATA_ID)
        X = dataset.data.features.to_numpy(dtype=float)
        y = dataset.data.targets.to_numpy(dtype=float).reshape(-1)

    X, y = _validate_regression_data(X, y, POWER_PLANT_DATASET_NAME)
    if not POWER_PLANT_CACHE_PATH.exists():
        POWER_PLANT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(POWER_PLANT_CACHE_PATH, X=X, y=y)
    return X, y


def _extract_electrical_grid_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Select continuous stability and remove the non-predictive p1 field."""
    features = dataset.data.features.drop(columns=["p1"], errors="raise")
    if "stab" not in dataset.data.targets:
        raise ValueError("UCI Electrical Grid data does not contain the 'stab' target")
    return (
        features.to_numpy(dtype=float),
        dataset.data.targets["stab"].to_numpy(dtype=float),
    )


@lru_cache(maxsize=1)
def load_electrical_grid_data() -> tuple[np.ndarray, np.ndarray]:
    """Load, validate, and cache UCI Electrical Grid Stability data."""
    if GRID_CACHE_PATH.exists():
        with np.load(GRID_CACHE_PATH) as cached:
            X = cached["X"]
            y = cached["y"]
    else:
        X, y = _extract_electrical_grid_arrays(fetch_ucirepo(id=GRID_UCI_DATA_ID))

    X, y = _validate_regression_data(X, y, GRID_DATASET_NAME)
    if not GRID_CACHE_PATH.exists():
        GRID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GRID_CACHE_PATH, X=X, y=y)
    return X, y


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load either shared real-world dataset by name."""
    loaders = {
        POWER_PLANT_DATASET_NAME: load_power_plant_data,
        GRID_DATASET_NAME: load_electrical_grid_data,
    }
    try:
        return loaders[dataset_name]()
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}") from error


def build_model_factories(hidden_size: int | None = None) -> dict[str, object]:
    """Build the seven shared model configurations."""

    def sparse_gp_factory(X_train, y_train, **kwargs):
        return SparseGPModel(
            X_train,
            y_train,
            learning_rate=GP_DKL_LEARNING_RATE,
            max_epochs=MODEL_MAX_EPOCHS,
            **kwargs,
        )

    factories: dict[str, object] = {
        "exact_gp": lambda X_train, y_train, **kwargs: ExactGPModel(
            X_train,
            y_train,
            learning_rate=GP_DKL_LEARNING_RATE,
            max_epochs=MODEL_MAX_EPOCHS,
            **kwargs,
        ),
    }
    for n_inducing in SPARSE_GP_INDUCING_POINTS:
        factories[f"sparse_gp_m{n_inducing}"] = ModelSpec(
            sparse_gp_factory, n_inducing=n_inducing
        )
    factories.update(
        {
            "deep_ensemble": lambda X_train, y_train, **kwargs: DeepEnsembleModel(
                X_train,
                y_train,
                n_members=DEEP_ENSEMBLE_MEMBERS,
                hidden_size=hidden_size,
                learning_rate=NEURAL_LEARNING_RATE,
                max_epochs=MODEL_MAX_EPOCHS,
                **kwargs,
            ),
            "mc_dropout": lambda X_train, y_train, **kwargs: MCDropoutModel(
                X_train,
                y_train,
                n_samples=MC_DROPOUT_SAMPLES,
                hidden_size=hidden_size,
                learning_rate=NEURAL_LEARNING_RATE,
                max_epochs=MODEL_MAX_EPOCHS,
                **kwargs,
            ),
            "dkl": lambda X_train, y_train, **kwargs: DKLModel(
                X_train,
                y_train,
                hidden_size=hidden_size,
                learning_rate=GP_DKL_LEARNING_RATE,
                max_epochs=MODEL_MAX_EPOCHS,
                **kwargs,
            ),
        }
    )
    return factories


__all__ = [
    "DATASET_NAMES",
    "GRID_DATASET_NAME",
    "N_SPLITS",
    "POWER_PLANT_DATASET_NAME",
    "UCI_DATA_IDS",
    "VALIDATION_RATIO",
    "build_model_factories",
    "load_dataset",
    "load_electrical_grid_data",
    "load_power_plant_data",
]
