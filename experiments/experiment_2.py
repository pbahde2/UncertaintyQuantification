"""Experiment 2: predictive uncertainty on synthetic and real OOD data."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execute_experiment import count_experiment_jobs, execute_experiment
from experiments.common import (
    N_SPLITS,
    build_model_factories,
    load_power_plant_data,
)


SYNTHETIC_DATASET_NAME = "synthetic_ood"
POWER_PLANT_DATASET_NAME = "power_plant_ood"
DATASET_NAMES = [SYNTHETIC_DATASET_NAME, POWER_PLANT_DATASET_NAME]

SYNTHETIC_TRAIN_SIZE = 100
POWER_PLANT_TRAIN_SIZE = 400
TRAIN_SIZES_BY_DATASET = {
    SYNTHETIC_DATASET_NAME: SYNTHETIC_TRAIN_SIZE,
    POWER_PLANT_DATASET_NAME: POWER_PLANT_TRAIN_SIZE,
}

SYNTHETIC_TEST_SIZE = 500
TRAIN_BOUND = 3.0
TEST_BOUND = 5.0
NOISE_STD = 0.2
SYNTHETIC_NN_HIDDEN_SIZE = 10

RESULT_PATH = ROOT / "results" / "tables" / "experiment_2_new.csv"
FIGURE_DIRECTORY = ROOT / "results" / "figures" / "experiment_2_new"

RANDOM_SEED = 0


def latent_function(x: np.ndarray) -> np.ndarray:
    """Return the noise-free synthetic regression function."""
    return (
        0.12 * x
        + 1.2 * np.exp(-0.5 * ((x + 2.1) / 0.7)**2)
        - 1.4 * np.exp(-0.5 * ((x + 0.7) / 0.65)**2)
        + 1.4 * np.exp(-0.5 * ((x - 0.7) / 0.65)**2)
        - 1.2 * np.exp(-0.5 * ((x - 2.1) / 0.7)**2)
    )


def synthetic_ood_dataset(
    dataset_name: str = SYNTHETIC_DATASET_NAME,
    random_seed: int = RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Generate one fixed training label budget and an evenly spaced test set."""
    if dataset_name != SYNTHETIC_DATASET_NAME:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    rng = np.random.default_rng(random_seed)
    X_train = rng.uniform(-TRAIN_BOUND, TRAIN_BOUND, size=(SYNTHETIC_TRAIN_SIZE, 1))
    X_test = np.linspace(-TEST_BOUND, TEST_BOUND, SYNTHETIC_TEST_SIZE).reshape(-1, 1)
    y_train = latent_function(X_train[:, 0]) + NOISE_STD * rng.normal(
        size=SYNTHETIC_TRAIN_SIZE
    )
    y_test = latent_function(X_test[:, 0]) + NOISE_STD * rng.normal(
        size=SYNTHETIC_TEST_SIZE
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_region": np.where(
            np.abs(X_test[:, 0]) <= TRAIN_BOUND, "id", "ood"
        ),
    }

def power_plant_id_ood_indices(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed middle-80% ID and two-tailed 20% OOD partition."""
    X = np.asarray(X, dtype=float)
    pc1_scores = PCA(n_components=1).fit_transform(
        StandardScaler().fit_transform(X)
    )[:, 0]
    ordered_indices = np.argsort(pc1_scores, kind="mergesort")
    tail_size = int(np.floor(0.10 * len(X)))
    ood_indices = np.concatenate(
        [ordered_indices[:tail_size], ordered_indices[-tail_size:]]
    )
    id_indices = ordered_indices[tail_size:-tail_size]
    return id_indices, ood_indices


def power_plant_ood_dataset(
    dataset_name: str = POWER_PLANT_DATASET_NAME,
    random_seed: int = RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Create a seed-specific ID train pool and fixed PCA-based OOD test region."""
    if dataset_name != POWER_PLANT_DATASET_NAME:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    X, y = load_power_plant_data()
    id_indices, ood_indices = power_plant_id_ood_indices(X)
    rng = np.random.default_rng(random_seed)
    train_indices = rng.choice(id_indices, size=POWER_PLANT_TRAIN_SIZE, replace=False)
    remaining_id_indices = np.setdiff1d(id_indices, train_indices, assume_unique=True)
    test_indices = np.concatenate([remaining_id_indices, ood_indices])

    return {
        "X_train": X[train_indices],
        "X_test": X[test_indices],
        "y_train": y[train_indices],
        "y_test": y[test_indices],
        "test_region": np.concatenate(
            [
                np.full(len(remaining_id_indices), "id"),
                np.full(len(ood_indices), "ood"),
            ]
        ),
    }


def experiment_2_dataset(
    dataset_name: str, random_seed: int = RANDOM_SEED
) -> dict[str, np.ndarray]:
    """Dispatch to the requested Experiment 2 dataset."""
    if dataset_name == SYNTHETIC_DATASET_NAME:
        return synthetic_ood_dataset(dataset_name, random_seed)
    if dataset_name == POWER_PLANT_DATASET_NAME:
        return power_plant_ood_dataset(dataset_name, random_seed)
    raise ValueError(f"Unsupported dataset: {dataset_name!r}")


def _model_factories_for(dataset_name: str) -> dict[str, object]:
    """Use a fixed neural width only for the one-dimensional synthetic dataset."""
    hidden_size = (
        SYNTHETIC_NN_HIDDEN_SIZE if dataset_name == SYNTHETIC_DATASET_NAME else None
    )
    return build_model_factories(hidden_size=hidden_size)


class PredictionCollector:
    """Collect synthetic predictions and save one aggregate plot per model."""

    def __init__(self, output_directory: Path = FIGURE_DIRECTORY) -> None:
        self.output_directory = output_directory
        self.grid = np.linspace(-TEST_BOUND, TEST_BOUND, SYNTHETIC_TEST_SIZE).reshape(
            -1, 1
        )
        self.predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self.training_sets: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(
        self,
        *,
        model: object,
        model_name: str,
        dataset_name: str = SYNTHETIC_DATASET_NAME,
        split_idx: int = 0,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        feature_scaler: object,
        **_: object,
    ) -> None:
        """Collect one synthetic seed's predictions and complete training subset."""
        if dataset_name != SYNTHETIC_DATASET_NAME:
            return
        mean, variance = model.predict(feature_scaler.transform(self.grid))
        self.predictions.setdefault(model_name, []).append(
            (np.asarray(mean), np.maximum(np.asarray(variance), 1e-12))
        )
        if X_train is not None and y_train is not None:
            self.training_sets.setdefault(split_idx, (X_train.copy(), y_train.copy()))

    def save_figures(self) -> list[Path]:
        """Write moment-matched 90% prediction intervals across seeds."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []
        for model_name, predictions in self.predictions.items():
            means = np.stack([mean for mean, _ in predictions])
            variances = np.stack([variance for _, variance in predictions])
            mixture_mean = np.mean(means, axis=0)
            mixture_variance = np.mean(variances + means**2, axis=0) - mixture_mean**2
            interval_radius = 1.6448536269514722 * np.sqrt(
                np.maximum(mixture_variance, 1e-12)
            )

            figure, axis = plt.subplots(figsize=(8.0, 4.6))
            figure.patch.set_facecolor("white")
            axis.set_facecolor("white")
            axis.plot(
                self.grid[:, 0], latent_function(self.grid[:, 0]), color="black",
                linewidth=1.7, label="True function", zorder=5,
            )
            axis.plot(
                self.grid[:, 0], mixture_mean, color="#910730", linewidth=1.8,
                label="Mean prediction across seeds", zorder=4,
            )
            axis.fill_between(
                self.grid[:, 0], mixture_mean - interval_radius,
                mixture_mean + interval_radius, color="#910730", alpha=0.16,
                label="Nominal 90% prediction interval", zorder=1,
            )
            axis.axvline(
                -TRAIN_BOUND, color="#143163", linestyle=(0, (4, 2)),
                linewidth=1.2, label="ID/OOD boundary", zorder=3,
            )
            axis.axvline(
                TRAIN_BOUND, color="#143163", linestyle=(0, (4, 2)),
                linewidth=1.2, zorder=3,
            )
            axis.set(
                xlim=(-TEST_BOUND, TEST_BOUND),
                xlabel="Input x", ylabel="Target y",
            )
            axis.xaxis.label.set_size(8.2)
            axis.yaxis.label.set_size(8.2)
            axis.tick_params(axis="both", labelsize=7.5, length=3)
            axis.grid(True, which="major", color="0.88", linewidth=0.55)
            axis.grid(False, which="minor")
            for spine in axis.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("0.35")
            handles, labels = axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            figure.legend(
                unique.values(), unique.keys(), loc="upper center",
                bbox_to_anchor=(0.5, 0.99), ncol=len(unique), frameon=False,
                columnspacing=1.0, handlelength=2.2, fontsize=7.5,
            )
            figure.subplots_adjust(
                top=0.87, bottom=0.14, left=0.10, right=0.985
            )

            output_path = self.output_directory / f"{model_name}.png"
            pdf_path = self.output_directory / f"{model_name}.pdf"
            figure.savefig(output_path, dpi=300, bbox_inches="tight")
            figure.savefig(pdf_path, bbox_inches="tight")
            plt.close(figure)
            output_paths.append(output_path)
        return output_paths


def _run_dataset(
    dataset_name: str, prediction_collector: PredictionCollector
) -> pd.DataFrame:
    """Run one dataset with its prescribed fitting-set size."""

    # Only collect synthetic predictions for one aggregate figure per model.
    callback = prediction_collector if dataset_name == SYNTHETIC_DATASET_NAME else None
    return execute_experiment(
        model_factories=_model_factories_for(dataset_name),
        dataset_names=[dataset_name],
        train_sizes=[TRAIN_SIZES_BY_DATASET[dataset_name]],
        n_splits=N_SPLITS,
        dataset_loader=experiment_2_dataset,
        test_region_key="test_region",
        fitted_model_callback=callback,
        verbose=True,
    )


def run_experiment_2() -> pd.DataFrame:
    """Run both OOD settings and save per-seed results and figure outputs."""
    total_jobs = sum(
        count_experiment_jobs(
            _model_factories_for(name),
            [TRAIN_SIZES_BY_DATASET[name]],
            n_splits=N_SPLITS,
        )
        for name in DATASET_NAMES
    )
    print(
        f"Datasets: {DATASET_NAMES}\n"
        f"Protocol: {N_SPLITS} seeds, with separate ID/OOD evaluation\n"
        f"Total model runs: {total_jobs}\n"
        f"Output: {RESULT_PATH}",
        flush=True,
    )

    # Collect synthetic predictions for one aggregate figure per model.
    prediction_collector = PredictionCollector()
    result_df = pd.concat(
        [_run_dataset(name, prediction_collector) for name in DATASET_NAMES],
        ignore_index=True,
    )
    figure_paths = prediction_collector.save_figures()

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULT_PATH, index=False)
    print(f"\nSaved per-seed results to: {RESULT_PATH}")
    print(f"Saved {len(figure_paths)} synthetic figures to: {FIGURE_DIRECTORY}")
    return result_df


if __name__ == "__main__":
    run_experiment_2()
