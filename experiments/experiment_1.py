"""Experiment 1: learning curves on three real-world regression datasets."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execute_experiment import count_experiment_jobs, execute_experiment
from experiments.common import (
    DATASET_NAMES,
    N_SPLITS,
    OPENML_DATA_IDS,
    UCI_DATA_IDS,
    build_model_factories,
    load_dataset,
)


TRAIN_FRACTION = 0.8 
TRAIN_SIZES = [50, 100, 200, 400, 800, 1_600, 3200]



OUTPUT_PATH = ROOT / "results" / "tables" / "experiment_1_lr_05_2.csv"


def benchmark_dataset(dataset_name: str, random_seed: int = 0):
    """Create one reproducible 80/20 partition for a benchmark dataset."""
    X, y = load_dataset(dataset_name)

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


def dataset_metadata() -> dict[str, dict[str, int | str | None]]:
    """Derive dataset dimensions and split sizes from the loaded data."""
    metadata = {}
    for dataset_name in DATASET_NAMES:
        X, _ = load_dataset(dataset_name)
        split = benchmark_dataset(dataset_name, random_seed=0)
        repository = "openml" if dataset_name in OPENML_DATA_IDS else "uci"
        metadata[dataset_name] = {
            "data_repository": repository,
            "data_id": (
                OPENML_DATA_IDS[dataset_name]
                if repository == "openml"
                else UCI_DATA_IDS[dataset_name]
            ),
            "total_observations": len(X),
            "n_features": X.shape[1],
            "train_pool_size": len(split["X_train"]),
            "test_size": len(split["X_test"]),
        }
    return metadata


def run_experiment_1() -> None:
    """Run Experiment 1 and write its dedicated result table."""
    model_factories = build_model_factories()
    metadata = dataset_metadata()
    dataset_lines = "\n".join(
        f"  {name}: {spec['total_observations']} rows, {spec['n_features']} features, "
        f"{spec['train_pool_size']} train-pool rows, {spec['test_size']} test rows"
        for name, spec in metadata.items()
    )
    print(
        f"Datasets:\n{dataset_lines}\n"
        f"Protocol: {N_SPLITS} splits per dataset\n"
        f"Training sizes: {TRAIN_SIZES}\n"
        f"Total model runs: "
        f"{count_experiment_jobs(model_factories, TRAIN_SIZES, len(DATASET_NAMES), N_SPLITS)}\n"
        f"Output: {OUTPUT_PATH}",
        flush=True,
    )

    result_df = execute_experiment(
        model_factories=model_factories,
        dataset_names=DATASET_NAMES,
        train_sizes=TRAIN_SIZES,
        n_splits=N_SPLITS,
        dataset_loader=benchmark_dataset,
        verbose=True,
    )

    result_df.insert(
        3,
        "train_pool_size",
        result_df["dataset"].map(
            {name: spec["train_pool_size"] for name, spec in metadata.items()}
        ),
    )
    result_df.insert(
        4,
        "test_size",
        result_df["dataset"].map(
            {name: spec["test_size"] for name, spec in metadata.items()}
        ),
    )
    result_df.insert(
        5,
        "data_repository",
        result_df["dataset"].map(
            {name: spec["data_repository"] for name, spec in metadata.items()}
        ),
    )
    result_df.insert(
        6,
        "data_id",
        result_df["dataset"].map(
            {name: spec["data_id"] for name, spec in metadata.items()}
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
    run_experiment_1()
