"""Run one uncertainty benchmark on UCI Power Plant and Protein Structure data."""

from __future__ import annotations

import io
import ssl
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

import certifi
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execute_experiment import ModelSpec, count_experiment_jobs, execute_experiment
from models.deep_ensemble import DeepEnsembleModel
from models.dkl import DKLModel
from models.exact_gp import ExactGPModel
from models.mc_dropout import MCDropoutModel
from models.sparse_gp import SparseGPModel


DATASET_NAME = "combined_cycle_power_plant"
UCI_DATA_ID = 294
TOTAL_OBSERVATIONS = 9_568
N_FEATURES = 4
PROTEIN_DATASET_NAME = "protein_tertiary_structure"
PROTEIN_UCI_DATA_ID = 265
PROTEIN_TOTAL_OBSERVATIONS = 45_730
PROTEIN_N_FEATURES = 9
TRAIN_FRACTION = 0.8
TRAIN_POOL_SIZE = 7_654
TEST_SIZE = 1_914
PROTEIN_TRAIN_POOL_SIZE = 36_584
PROTEIN_TEST_SIZE = 9_146
TRAIN_SIZES = [50, 100, 200, 400, 800, 1_600]
N_SPLITS = 20
CACHE_PATH = ROOT / ".cache" / "uci" / "power_plant_294.npz"
PROTEIN_CACHE_PATH = ROOT / ".cache" / "uci" / "protein_structure_265.npz"
PROTEIN_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/265/"
    "physicochemical+properties+of+protein+tertiary+structure.zip"
)

DATASET_SPECS = {
    DATASET_NAME: {
        "uci_data_id": UCI_DATA_ID,
        "total_observations": TOTAL_OBSERVATIONS,
        "n_features": N_FEATURES,
        "train_pool_size": TRAIN_POOL_SIZE,
        "test_size": TEST_SIZE,
    },
    PROTEIN_DATASET_NAME: {
        "uci_data_id": PROTEIN_UCI_DATA_ID,
        "total_observations": PROTEIN_TOTAL_OBSERVATIONS,
        "n_features": PROTEIN_N_FEATURES,
        "train_pool_size": PROTEIN_TRAIN_POOL_SIZE,
        "test_size": PROTEIN_TEST_SIZE,
    },
}


def _load_verified_uci_data(
    *, dataset_name: str, uci_data_id: int, expected_rows: int,
    expected_features: int, cache_path: Path,
    raw_loader: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and cache one UCI regression dataset after validating its arrays."""
    if cache_path.exists():
        with np.load(cache_path) as cached:
            X = cached["X"]
            y = cached["y"]
    else:
        if raw_loader is None:
            dataset = fetch_ucirepo(id=uci_data_id)
            X = dataset.data.features.to_numpy(dtype=float)
            y = dataset.data.targets.to_numpy(dtype=float).reshape(-1)
        else:
            X, y = raw_loader()

    expected_X_shape = (expected_rows, expected_features)
    expected_y_shape = (expected_rows,)
    if X.shape != expected_X_shape or y.shape != expected_y_shape:
        raise ValueError(
            f"Unexpected {dataset_name} shape: X={X.shape}, y={y.shape}; "
            f"expected {expected_X_shape} and {expected_y_shape}."
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{dataset_name} contains non-finite values.")

    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def _download_protein_structure_data() -> tuple[np.ndarray, np.ndarray]:
    """Download dataset 265 directly because ucimlrepo does not expose its import API."""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(PROTEIN_ARCHIVE_URL, context=ssl_context) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_name = next(
            (name for name in archive.namelist() if name.lower().endswith("casp.csv")),
            None,
        )
        if csv_name is None:
            raise ValueError("The UCI Protein Structure archive does not contain CASP.csv.")
        with archive.open(csv_name) as csv_file:
            values = np.loadtxt(csv_file, delimiter=",", skiprows=1, dtype=float)
    # CASP.csv stores the RMSD regression target first, followed by features F1-F9.
    return values[:, 1:], values[:, 0]


@lru_cache(maxsize=1)
def load_power_plant_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the UCI dataset, caching the verified arrays between runs."""
    return _load_verified_uci_data(
        dataset_name=DATASET_NAME,
        uci_data_id=UCI_DATA_ID,
        expected_rows=TOTAL_OBSERVATIONS,
        expected_features=N_FEATURES,
        cache_path=CACHE_PATH,
    )


@lru_cache(maxsize=1)
def load_protein_structure_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the UCI Protein Structure dataset and cache its verified arrays."""
    return _load_verified_uci_data(
        dataset_name=PROTEIN_DATASET_NAME,
        uci_data_id=PROTEIN_UCI_DATA_ID,
        expected_rows=PROTEIN_TOTAL_OBSERVATIONS,
        expected_features=PROTEIN_N_FEATURES,
        cache_path=PROTEIN_CACHE_PATH,
        raw_loader=_download_protein_structure_data,
    )


def power_plant_dataset(dataset_name: str = DATASET_NAME, random_seed: int = 0):
    """Create one reproducible 80/20 partition for either benchmark dataset."""
    if dataset_name == DATASET_NAME:
        X, y = load_power_plant_data()
    elif dataset_name == PROTEIN_DATASET_NAME:
        X, y = load_protein_structure_data()
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


def power_plant_experiment() -> None:
    """Run both datasets through seven configurations, six sizes, and twenty splits."""
    model_factories = {
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
        f"{count_experiment_jobs(model_factories, TRAIN_SIZES, len(dataset_names), N_SPLITS)}",
        flush=True,
    )

    result_df = execute_experiment(
        model_factories=model_factories,
        dataset_names=dataset_names,
        train_sizes=TRAIN_SIZES,
        n_splits=N_SPLITS,
        dataset_loader=power_plant_dataset,
        verbose=True,
    )

    result_df.insert(
        3, "train_pool_size",
        result_df["dataset"].map(
            {name: spec["train_pool_size"] for name, spec in DATASET_SPECS.items()}
        ),
    )
    result_df.insert(
        4, "test_size",
        result_df["dataset"].map(
            {name: spec["test_size"] for name, spec in DATASET_SPECS.items()}
        ),
    )
    result_df.insert(
        5, "uci_data_id",
        result_df["dataset"].map(
            {name: spec["uci_data_id"] for name, spec in DATASET_SPECS.items()}
        ),
    )

    output_path = ROOT / "results" / "tables" / "power_plant_experiment.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    summary = (
        result_df.groupby(["dataset", "model", "train_size"], as_index=False)[
            ["rmse", "nll", "crps", "coverage_90"]
        ]
        .mean()
        .sort_values(["dataset", "model", "train_size"])
    )
    print(summary)
    print(f"\nSaved benchmark table to: {output_path}")


if __name__ == "__main__":
    power_plant_experiment()
