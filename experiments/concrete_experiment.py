"""Run the uncertainty benchmark on UCI Concrete Compressive Strength.

The dataset contains 1,030 observations. To support the requested maximum training
size of 1,000, every repetition reserves 30 observations for testing and uses the
remaining 1,000 as a nested training pool. The train/test partition changes with the
split seed but is shared by every model and training size within that split.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execute_experiment import execute_experiment
from models.deep_ensemble import DeepEnsembleModel
from models.dkl import DKLModel
from models.exact_gp import ExactGPModel
from models.mc_dropout import MCDropoutModel
from models.sparse_gp import SparseGPModel


DATASET_NAME = "concrete_compressive_strength"
OPENML_DATA_ID = 44959
TOTAL_OBSERVATIONS = 1_030
TRAIN_POOL_SIZE = 1_000
TEST_SIZE = TOTAL_OBSERVATIONS - TRAIN_POOL_SIZE
TRAIN_SIZES = [50, 100, 200, 400, 800, 1_000]
N_SPLITS = 10


@lru_cache(maxsize=1)
def load_concrete_data() -> tuple[np.ndarray, np.ndarray]:
    """Download the verified OpenML copy of the UCI Concrete dataset once."""
    X, y = fetch_openml(
        data_id=OPENML_DATA_ID,
        data_home=ROOT / ".cache" / "scikit_learn",
        as_frame=False,
        return_X_y=True,
        parser="auto",
    )
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.shape != (TOTAL_OBSERVATIONS, 8) or y.shape != (TOTAL_OBSERVATIONS,):
        raise ValueError(
            "Unexpected Concrete dataset shape: "
            f"X={X.shape}, y={y.shape}; expected (1030, 8) and (1030,)."
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Concrete dataset contains non-finite values.")
    return X, y


def concrete_dataset(dataset_name: str = DATASET_NAME, random_seed: int = 0):
    """Create one reproducible 1,000/30 train/test partition.

    A fixed-size split is necessary here: the usual 80/20 split would leave only
    824 observations for training and could not evaluate the requested size 1,000.
    """
    if dataset_name != DATASET_NAME:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    X, y = load_concrete_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=TRAIN_POOL_SIZE,
        test_size=TEST_SIZE,
        random_state=random_seed,
        shuffle=True,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def concrete_experiment() -> None:
    """Run all seven model configurations over six sizes and ten splits."""
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
        f"Dataset: {DATASET_NAME} ({TOTAL_OBSERVATIONS} rows, 8 features)\n"
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
        dataset_loader=concrete_dataset,
        verbose=True,
    )

    # Record the nonstandard 1,000/30 split directly in the exported result table.
    result_df.insert(3, "train_pool_size", TRAIN_POOL_SIZE)
    result_df.insert(4, "test_size", TEST_SIZE)
    result_df.insert(5, "openml_data_id", OPENML_DATA_ID)

    output_path = ROOT / "results" / "tables" / "concrete_experiment.csv"
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
    concrete_experiment()
