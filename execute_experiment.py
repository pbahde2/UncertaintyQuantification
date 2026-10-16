"""Experiment execution runner for comparing uncertainty-aware regression methods.

This module is the central orchestration point for the project. It loads data, creates
subsets of different sizes, trains multiple models, evaluates them under the same metrics,
then returns the resulting experiment table.

The idea is to keep the research protocol consistent across all methods: same datasets,
standardized features, matched train/test splits, and identical statistical reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from data_preparation import (
    nested_training_subsets,
    prepare_dataset_for_benchmark,
    split_train_validation,
    standardize_features,
)
from metrics import compute_regression_metrics


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    This class defines the grid over datasets, training sizes, repetitions, and model
    functions used in the experimental loop.
    """

    dataset_names: Iterable[str]
    train_sizes: Iterable[int]
    n_splits: int = 20
    random_seed: int = 0
    test_fraction: float = 0.2
    validation_fraction: float = 0.2


def _format_duration(seconds: float) -> str:
    """Format a duration compactly for progress and ETA messages."""
    seconds = float(seconds)
    if not np.isfinite(seconds):
        return "n/a"
    seconds = max(0.0, seconds)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m"


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
    # create and train the model using the provided factory function
    model = model_factory(X_train, y_train, **kwargs)
    # evaluate the model on the test set and compute metrics
    mean, variance = model.predict(X_test)
    std = np.sqrt(np.maximum(variance, 1e-12))
    metrics = compute_regression_metrics(y_test, mean, std)

    return {
        "model": model_name,
        "rmse": metrics["rmse"],
        "nll": metrics["nll"],
        "crps": metrics["crps"],
        "coverage_90": metrics["coverage_90"],
    }


def evaluate_trained_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Evaluate an already-trained model whose prediction API returns variance."""
    # Time prediction only; metric computation below is intentionally excluded.
    inference_start = time.perf_counter()
    mean, variance = model.predict(X_test)
    inference_time_seconds = time.perf_counter() - inference_start
    std = np.sqrt(np.maximum(variance, 1e-12))
    metrics = compute_regression_metrics(y_test, mean, std)
    return {
        "model": model_name,
        "inference_time_seconds": inference_time_seconds,
        **metrics,
    }


def execute_experiment(
    model_factories: Dict[str, Callable[..., Any]],
    dataset_names: Iterable[str],
    train_sizes: Iterable[int],
    n_splits: int = 20,
    validation_fraction: float = 0.2,
    dataset_loader: Optional[Callable[[str], Dict[str, np.ndarray]]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the experiment across datasets, training sizes, and model families.

    This is the main experimental loop. It performs the repeated train/test protocol and
    builds a tabular result set that can later be exported to CSV or plotted as learning
    curves. Training sizes are nested within a split. Each subset is partitioned into model-fit
    and validation observations, and feature scaling is fitted only on the model-fit partition.
    A custom ``dataset_loader`` can be supplied for synthetic or custom datasets.
    """
    rows: List[dict] = []
    loader = dataset_loader or prepare_dataset_for_benchmark
    dataset_names = list(dataset_names)
    train_sizes = list(train_sizes)
    total_jobs = len(dataset_names) * n_splits * len(train_sizes) * len(model_factories)
    completed_jobs = 0
    experiment_start = time.perf_counter()
    if verbose:
        print(f"Starting {total_jobs} model runs...", flush=True)

    # For every dataset
    for dataset_name in dataset_names:
        #For every split of the dataset
        for split_idx in range(n_splits):
            data = loader(dataset_name, random_seed=split_idx)
            X_train, X_test, y_train, y_test = (
                data["X_train"],
                data["X_test"],
                data["y_train"],
                data["y_test"],
            )
            # Create nested training subsets for the current split and train sizes
            subsets = nested_training_subsets(
                X_train, y_train, train_sizes, random_seed=split_idx
            )
            # For every training size, train the model and evaluate on the test set
            for train_size, (X_sub, y_sub) in subsets.items():
                # Hold out validation data only to select how many epochs the final model trains.
                X_fit, X_val, y_fit, y_val = split_train_validation(
                    X_sub, y_sub, validation_fraction=validation_fraction
                )
                # Scale the selection split without exposing validation observations.
                X_fit_scaled, X_test_scaled, scaler = standardize_features(X_fit, X_test)
                X_val_scaled = scaler.transform(X_val) if len(X_val) else X_val.copy()
                # The final model uses all requested observations and their preprocessing statistics.
                X_full_scaled, X_final_test_scaled, _ = standardize_features(X_sub, X_test)

                for model_name, model_factory in model_factories.items():
                    # Stage 1: early-stop on validation NLL to determine the training duration.
                    job_number = completed_jobs + 1
                    if verbose:
                        print(
                            f"[{job_number}/{total_jobs}] {dataset_name} | split={split_idx} | "
                            f"train_size={train_size} | {model_name}: selecting epoch...",
                            flush=True,
                        )
                    selection_start = time.perf_counter()
                    model = model_factory(
                        X_fit_scaled,
                        y_fit,
                        X_val=X_val_scaled,
                        y_val=y_val,
                        random_seed=split_idx,
                    )
                    # Save the selected duration before the model is retrained on all observations.
                    member_epochs = getattr(model, "selected_epochs_", None)
                    if member_epochs is not None:
                        selected_epoch = float(np.mean(member_epochs))
                    else:
                        selected_epoch = float(getattr(model, "best_epoch_", np.nan))
                    selection_time_seconds = time.perf_counter() - selection_start
                    if verbose:
                        print(
                            f"    selected_epoch={selected_epoch:g} after "
                            f"{_format_duration(selection_time_seconds)}; final training...",
                            flush=True,
                        )

                    if hasattr(model, "refit_full"):
                        # Stage 2: reinitialize and fit on all train_size observations for that duration.
                        # Start timing only now, excluding validation-based epoch selection.
                        training_start = time.perf_counter()
                        model.refit_full(X_full_scaled, y_sub)
                        final_training_time_seconds = time.perf_counter() - training_start
                        evaluation_X = X_final_test_scaled
                    else:
                        # Backward-compatible path for external models without refitting support.
                        final_training_time_seconds = np.nan
                        evaluation_X = X_test_scaled
                    # For SVGP rows, record the post-cap count used by the final full-data model.
                    effective_inducing = float(getattr(model, "n_inducing_", np.nan))
                    result = evaluate_trained_model(model, evaluation_X, y_test, model_name)
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "split": split_idx,
                            "train_size": train_size,
                            "selected_epoch": selected_epoch,
                            "n_inducing": effective_inducing,
                            "final_training_time_seconds": final_training_time_seconds,
                            **result,
                        }
                    )
                    completed_jobs += 1
                    if verbose:
                        elapsed = time.perf_counter() - experiment_start
                        eta = elapsed / completed_jobs * (total_jobs - completed_jobs)
                        print(
                            f"    done | train={_format_duration(final_training_time_seconds)} | "
                            f"infer={_format_duration(result['inference_time_seconds'])} | "
                            f"elapsed={_format_duration(elapsed)} | ETA≈{_format_duration(eta)}",
                            flush=True,
                        )

    if verbose:
        elapsed = time.perf_counter() - experiment_start
        print(f"Finished {completed_jobs} model runs in {_format_duration(elapsed)}.", flush=True)
    return pd.DataFrame(rows)


__all__ = [
    "ExperimentConfig",
    "run_single_experiment",
    "evaluate_trained_model",
    "execute_experiment",
]
