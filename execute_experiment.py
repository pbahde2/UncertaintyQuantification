"""Shared orchestration for both uncertainty-quantification experiments."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from metrics import compute_regression_metrics


DEFAULT_INFERENCE_WARMUP_RUNS = 3
DEFAULT_INFERENCE_TIMING_REPEATS = 20

ModelFactory = Callable[..., Any]
DatasetLoader = Callable[..., dict[str, np.ndarray]]
FittedModelCallback = Callable[..., None]


@dataclass(frozen=True)
class ModelSpec:
    """A model factory with optional sparse-GP scheduling metadata."""

    factory: ModelFactory
    n_inducing: int | None = None

    def __post_init__(self) -> None:
        if self.n_inducing is not None and self.n_inducing <= 0:
            raise ValueError("n_inducing must be a positive integer")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.n_inducing is not None:
            kwargs["n_inducing"] = self.n_inducing
        return self.factory(*args, **kwargs)

    def supports_train_size(self, train_size: int) -> bool:
        return self.n_inducing is None or train_size > self.n_inducing


def _supports_train_size(model_factory: ModelFactory, train_size: int) -> bool:
    supports = getattr(model_factory, "supports_train_size", None)
    return supports is None or bool(supports(train_size))


def count_experiment_jobs(
    model_factories: Mapping[str, ModelFactory],
    train_sizes: Iterable[int],
    n_datasets: int = 1,
    n_splits: int = 1,
) -> int:
    """Count eligible model fits in an experiment."""
    runs_per_split = sum(
        _supports_train_size(factory, train_size)
        for train_size in train_sizes
        for factory in model_factories.values()
    )
    return n_datasets * n_splits * runs_per_split


def _nested_training_subsets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_sizes: Iterable[int],
    random_seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Create nested subsets representing each model's complete label budget."""
    sizes = [int(size) for size in train_sizes]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("train_sizes must contain positive integers")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have equal length")

    max_train_size = max(sizes)
    if max_train_size > len(X_train):
        raise ValueError(
            f"requested {max_train_size} training observations, but only "
            f"{len(X_train)} are available"
        )

    order = np.random.default_rng(random_seed).permutation(len(X_train))
    training_indices = order[:max_train_size]
    return {
        size: (X_train[training_indices[:size]], y_train[training_indices[:size]])
        for size in sizes
    }


def _predict_and_time(
    model: Any,
    X_test: np.ndarray,
    warmup_runs: int,
    timing_repeats: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return one scored prediction and a stable median inference time."""
    if warmup_runs < 1 or timing_repeats < 1:
        raise ValueError("inference warm-up runs and timing repeats must be positive")

    mean, variance = model.predict(X_test)
    for _ in range(warmup_runs - 1):
        model.predict(X_test)

    durations = []
    for _ in range(timing_repeats):
        start = time.perf_counter()
        model.predict(X_test)
        durations.append(time.perf_counter() - start)

    std = np.sqrt(np.maximum(variance, 1e-12))
    return np.asarray(mean), std, float(np.median(durations))


def _evaluation_rows(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    region_labels: np.ndarray | None,
    warmup_runs: int,
    timing_repeats: int,
) -> list[dict[str, Any]]:
    """Evaluate one prediction globally or separately for each ID/OOD region."""
    mean, std, inference_time = _predict_and_time(
        model, X_test, warmup_runs, timing_repeats
    )
    base = {
        "model": model_name,
        "inference_time_seconds": inference_time,
    }
    if region_labels is None:
        return [{**base, **compute_regression_metrics(y_test, mean, std)}]

    region_labels = np.asarray(region_labels)
    if len(region_labels) != len(y_test):
        raise ValueError("test region labels must match the test-set length")
    rows = []
    for region in np.unique(region_labels):
        mask = region_labels == region
        rows.append(
            {
                **base,
                "evaluation_region": str(region),
                "n_evaluation": int(mask.sum()),
                "mean_predictive_std": float(std[mask].mean()),
                **compute_regression_metrics(y_test[mask], mean[mask], std[mask]),
            }
        )
    return rows


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m"


def execute_experiment(
    *,
    model_factories: Mapping[str, ModelFactory],
    dataset_names: Iterable[str],
    train_sizes: Iterable[int],
    dataset_loader: DatasetLoader,
    n_splits: int = 20,
    test_region_key: str | None = None,
    fitted_model_callback: FittedModelCallback | None = None,
    verbose: bool = False,
    inference_warmup_runs: int = DEFAULT_INFERENCE_WARMUP_RUNS,
    inference_timing_repeats: int = DEFAULT_INFERENCE_TIMING_REPEATS,
) -> pd.DataFrame:
    """Train and evaluate every eligible model under the shared protocol."""
    dataset_names = list(dataset_names)
    train_sizes = list(train_sizes)
    total_jobs = count_experiment_jobs(
        model_factories, train_sizes, len(dataset_names), n_splits
    )
    rows: list[dict[str, Any]] = []
    completed_jobs = 0
    experiment_start = time.perf_counter()

    if verbose:
        print(
            f"Starting {total_jobs} model runs...\n"
            f"Inference timing: median of {inference_timing_repeats} calls after "
            f"{inference_warmup_runs} warm-up calls.",
            flush=True,
        )

    # per dataset,
    for dataset_name in dataset_names:
        #per split,
        for split_idx in range(n_splits):
            data = dataset_loader(dataset_name, random_seed=split_idx)
            X_train = np.asarray(data["X_train"])
            X_test = np.asarray(data["X_test"])
            y_train = np.asarray(data["y_train"])
            y_test = np.asarray(data["y_test"])
            region_labels = (
                np.asarray(data[test_region_key])
                if test_region_key is not None
                else None
            )
            # Draw nested subsets, each representing the complete label budget.
            subsets = _nested_training_subsets(
                X_train, y_train, train_sizes, split_idx
            )
            #per train size
            for train_size, (X_train_subset, y_train_subset) in subsets.items():
                # Scale features from the complete training subset.
                scaler = StandardScaler().fit(X_train_subset)
                X_train_scaled = scaler.transform(X_train_subset)
                X_test_scaled = scaler.transform(X_test)

                # per model
                for model_name, model_factory in model_factories.items():
                    if not _supports_train_size(model_factory, train_size):
                        continue
                    if verbose:
                        print(
                            f"[{completed_jobs + 1}/{total_jobs}] {dataset_name} | "
                            f"split={split_idx} | train_size={train_size} | "
                            f"{model_name}: training...",
                            flush=True,
                        )
                    # timing the training process
                    training_start = time.perf_counter()
                    # train model
                    model = model_factory(
                        X_train_scaled,
                        y_train_subset,
                        random_seed=split_idx,
                    )
                    #stop training timer
                    training_time = time.perf_counter() - training_start
                    member_epochs = getattr(model, "selected_epochs_", None)
                    selected_epoch = float(
                        np.mean(member_epochs)
                        if member_epochs is not None
                        else getattr(model, "best_epoch_", np.nan)
                    )
                    optimization_steps = float(
                        getattr(model, "optimization_steps_", np.nan)
                    )
                    optimization_epochs = float(
                        getattr(model, "optimization_epochs_", np.nan)
                    )
                    optimization_status = getattr(model, "optimization_status_", None)

                    #evaluate the model on the test set and compute metrics
                    results = _evaluation_rows(
                        model,
                        model_name,
                        X_test_scaled,
                        y_test,
                        region_labels,
                        inference_warmup_runs,
                        inference_timing_repeats,
                    )
                    for result in results:
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "split": split_idx,
                                "train_size": train_size,
                                "objective_train_size": train_size,
                                "validation_size": 0,
                                "selected_epoch": selected_epoch,
                                "optimization_epochs": optimization_epochs,
                                "optimization_steps": optimization_steps,
                                "optimization_status": optimization_status,
                                "n_inducing": float(
                                    getattr(model, "n_inducing_", np.nan)
                                ),
                                "final_training_time_seconds": training_time,
                                **result,
                            }
                        )
                    # used to create the figures for the 1D test case
                    if fitted_model_callback is not None:
                        fitted_model_callback(
                            model=model,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            split_idx=split_idx,
                            train_size=train_size,
                            X_train=X_train_subset,
                            y_train=y_train_subset,
                            feature_scaler=scaler,
                        )

                    completed_jobs += 1
                    if verbose:
                        elapsed = time.perf_counter() - experiment_start
                        eta = elapsed / completed_jobs * (total_jobs - completed_jobs)
                        stopping_parts = []
                        if np.isfinite(optimization_epochs):
                            stopping_parts.append(f"epochs={optimization_epochs:g}")
                        if np.isfinite(selected_epoch):
                            stopping_parts.append(f"selected_epoch={selected_epoch:g}")
                        if not stopping_parts:
                            stopping_parts.append(
                                f"optimization_status={optimization_status}"
                            )
                        stopping_detail = " | ".join(stopping_parts)
                        print(
                            f"    {stopping_detail} | "
                            f"train={_format_duration(training_time)} | "
                            f"infer={_format_duration(results[0]['inference_time_seconds'])} | "
                            f"elapsed={_format_duration(elapsed)} | "
                            f"ETA≈{_format_duration(eta)}",
                            flush=True,
                        )

    if verbose:
        elapsed = time.perf_counter() - experiment_start
        print(
            f"Finished {completed_jobs} model runs in {_format_duration(elapsed)}.",
            flush=True,
        )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_INFERENCE_TIMING_REPEATS",
    "DEFAULT_INFERENCE_WARMUP_RUNS",
    "ModelSpec",
    "count_experiment_jobs",
    "execute_experiment",
]
