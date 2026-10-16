"""Run the uncertainty benchmark on UCI Combined Cycle Power Plant.

The dataset contains 9,568 observations with four continuous ambient features and
one continuous power-output target. Each repetition uses a reproducible 80/20
train/test split. Training sizes are nested inside the resulting training pool.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execute_experiment import execute_experiment
from models.deep_ensemble import DeepEnsembleModel
from models.dkl import DKLModel
from models.exact_gp import ExactGPModel
from models.mc_dropout import MCDropoutModel
from models.sparse_gp import SparseGPModel


DATASET_NAME = "combined_cycle_power_plant"
UCI_DATA_ID = 294
TOTAL_OBSERVATIONS = 9_568
N_FEATURES = 4
TRAIN_FRACTION = 0.8
TRAIN_POOL_SIZE = 7_654
TEST_SIZE = 1_914
TRAIN_SIZES = [50, 100, 200, 400, 800, 1_600]
N_SPLITS = 20
CACHE_PATH = ROOT / ".cache" / "uci" / "power_plant_294.npz"


@lru_cache(maxsize=1)
def load_power_plant_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the UCI dataset, caching the verified arrays between runs."""
    if CACHE_PATH.exists():
        with np.load(CACHE_PATH) as cached:
            X = cached["X"]
            y = cached["y"]
    else:
        dataset = fetch_ucirepo(id=UCI_DATA_ID)
        X = dataset.data.features.to_numpy(dtype=float)
        y = dataset.data.targets.to_numpy(dtype=float).reshape(-1)

    expected_X_shape = (TOTAL_OBSERVATIONS, N_FEATURES)
    expected_y_shape = (TOTAL_OBSERVATIONS,)
    if X.shape != expected_X_shape or y.shape != expected_y_shape:
        raise ValueError(
            "Unexpected power-plant dataset shape: "
            f"X={X.shape}, y={y.shape}; expected {expected_X_shape} and "
            f"{expected_y_shape}."
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Power-plant dataset contains non-finite values.")

    if not CACHE_PATH.exists():
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_PATH, X=X, y=y)
    return X, y


def power_plant_dataset(dataset_name: str = DATASET_NAME, random_seed: int = 0):
    """Create one reproducible 80/20 train/test partition."""
    if dataset_name != DATASET_NAME:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    X, y = load_power_plant_data()
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


def power_plant_experiment() -> None:
    """Run seven model configurations over six sizes and twenty splits."""
    model_factories = {
        "exact_gp": lambda X_train, y_train, **kwargs: ExactGPModel(
            X_train, y_train, **kwargs
        ),
        "sparse_gp_m16": lambda X_train, y_train, **kwargs: SparseGPModel(
            X_train, y_train, n_inducing=16, **kwargs
        ),
        "sparse_gp_m64": lambda X_train, y_train, **kwargs: SparseGPModel(
            X_train, y_train, n_inducing=64, **kwargs
        ),
        "sparse_gp_m256": lambda X_train, y_train, **kwargs: SparseGPModel(
            X_train, y_train, n_inducing=256, **kwargs
        ),
        "deep_ensemble": lambda X_train, y_train, **kwargs: DeepEnsembleModel(
            X_train, y_train, n_members=5, **kwargs
        ),
        "mc_dropout": lambda X_train, y_train, **kwargs: MCDropoutModel(
            X_train, y_train, n_samples=25, **kwargs
        ),
        "dkl": lambda X_train, y_train, **kwargs: DKLModel(
            X_train, y_train, **kwargs
        ),
    }

    print(
        f"Dataset: {DATASET_NAME} "
        f"({TOTAL_OBSERVATIONS} rows, {N_FEATURES} features)\n"
        f"Protocol: {N_SPLITS} splits, {TRAIN_POOL_SIZE} training-pool rows and "
        f"{TEST_SIZE} held-out test rows per split\n"
        f"Training sizes: {TRAIN_SIZES}\n"
        f"Total model runs: "
        f"{N_SPLITS * len(TRAIN_SIZES) * len(model_factories)}",
        flush=True,
    )

    result_df = execute_experiment(
        model_factories=model_factories,
        dataset_names=[DATASET_NAME],
        train_sizes=TRAIN_SIZES,
        n_splits=N_SPLITS,
        dataset_loader=power_plant_dataset,
        verbose=True,
    )

    result_df.insert(3, "train_pool_size", TRAIN_POOL_SIZE)
    result_df.insert(4, "test_size", TEST_SIZE)
    result_df.insert(5, "uci_data_id", UCI_DATA_ID)

    output_path = ROOT / "results" / "tables" / "power_plant_experiment.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    summary = (
        result_df.groupby(["model", "train_size"], as_index=False)[
            ["rmse", "nll", "crps", "coverage_90"]
        ]
        .mean()
        .sort_values(["model", "train_size"])
    )
    print(summary)
    print(f"\nSaved benchmark table to: {output_path}")


if __name__ == "__main__":
    power_plant_experiment()
