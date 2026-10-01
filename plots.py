"""Visualization helpers for the uncertainty benchmark.

This module collects all plotting code in one place so that the benchmark results can be
translated into learning curves, calibration plots, and 1-D extrapolation visualizations.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_learning_curve(results: pd.DataFrame, metric: str, dataset: str | None = None) -> None:
    """Plot a learning curve for a specific metric.

    Parameters
    ----------
    results : pd.DataFrame
        Benchmark output table containing at least:model, dataset, train_size, and the
        metric column.
    metric : str
        Metric to visualize, e.g. 'rmse', 'nll', 'crps', or 'coverage_90'.
    dataset : str | None, optional
        Restrict the plot to a single dataset if supplied.
    """
    subset = results if dataset is None else results[results["dataset"] == dataset]
    grouped = subset.groupby(["train_size", "model"], as_index=False)[metric].mean()

    for model_name, group in grouped.groupby("model"):
        plt.plot(group["train_size"], group[metric], marker="o", label=model_name)

    plt.xlabel("Training set size")
    plt.ylabel(metric)
    plt.title(f"Learning curve for {metric}" if dataset is None else f"{dataset}: {metric}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_coverage_vs_nominal(results: pd.DataFrame, dataset: str | None = None) -> None:
    """Plot empirical coverage against nominal coverage.

    This helps assess whether interval calibration is close to the nominal 90% level, or
    whether a model is overconfident or underconfident.
    """
    subset = results if dataset is None else results[results["dataset"] == dataset]
    for model_name, group in subset.groupby("model"):
        coverage = group["coverage_90"].mean()
        plt.axhline(coverage, label=f"{model_name}: {coverage:.2f}")

    plt.axhline(0.9, color="black", linestyle="--", label="nominal 90%")
    plt.ylim(0, 1.1)
    plt.ylabel("Empirical coverage")
    plt.title("Coverage comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


__all__ = ["plot_learning_curve", "plot_coverage_vs_nominal"]
