"""Evaluation metrics for regression uncertainty methods.

This module centralizes the metrics used in the benchmark. The project evaluates each
model using both point-prediction quality and uncertainty calibration, which is why we
include RMSE, NLL, CRPS, and coverage metrics.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, logsumexp, ndtr, ndtri


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the root mean squared error.

    RMSE measures the average magnitude of predictive errors and is the most common
    point-prediction score in regression benchmarks.
    """
    errors = y_true - y_pred
    return float(np.sqrt(np.mean(errors**2)))


def negative_log_likelihood(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Compute the Gaussian negative log-likelihood.

    This score evaluates both the predictive mean and the uncertainty calibration. A model
    with well-calibrated uncertainty should achieve a lower NLL than a model with over- or
    under-confident predictions.
    """
    std = np.asarray(std, dtype=float)
    std = np.clip(std, 1e-6, None)
    residual = y_true - mean
    loss = 0.5 * np.log(2 * np.pi * std**2) + 0.5 * (residual / std) ** 2
    return float(np.mean(loss))


def interval_coverage(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, level: float = 0.9) -> float:
    """Compute the empirical coverage of a predictive interval.

    Parameters
    ----------
    y_true : np.ndarray
        Observed target values.
    mean : np.ndarray
        Predictive mean.
    std : np.ndarray
        Predictive standard deviation.
    level : float, optional
        Coverage level, e.g. 0.9 for the nominal 90% interval.

    Returns
    -------
    float
        Fraction of true targets lying inside the nominal interval.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    z = float(ndtri((1.0 + level) / 2.0))
    lower = mean - z * std
    upper = mean + z * std
    covered = np.logical_and(y_true >= lower, y_true <= upper)
    return float(np.mean(covered))


def crps_gaussian(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Approximate the CRPS for a Gaussian predictive distribution.

    CRPS is a proper scoring rule that rewards both sharpness and calibration. Lower is
    better, and it is often more informative than RMSE for probabilistic forecasts.
    """
    std = np.asarray(std, dtype=float)
    std = np.clip(std, 1e-6, None)

    # Closed-form CRPS for Gaussian predictive distributions.
    # This implementation is based on the standard expression for N(mu, sigma^2).
    z = (y_true - mean) / std
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2)))
    pdf = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi))

    crps = std * (
        z * (2 * cdf - 1)
        + 2 * pdf
        - 1 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None = None,
) -> dict:
    """Compute a standard set of regression and uncertainty metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values.
    y_pred : np.ndarray
        Predictive mean.
    y_std : np.ndarray | None, optional
        Predictive standard deviation for uncertainty-aware metrics.

    Returns
    -------
    dict
        Dictionary with keys: 'rmse', 'nll', 'crps', 'coverage_90'.
    """
    metrics = {
        "rmse": rmse(y_true, y_pred),
    }

    if y_std is not None:
        metrics["nll"] = negative_log_likelihood(y_true, y_pred, y_std)
        metrics["crps"] = crps_gaussian(y_true, y_pred, y_std)
        metrics["coverage_90"] = interval_coverage(y_true, y_pred, y_std, level=0.9)
    else:
        metrics["nll"] = np.nan
        metrics["crps"] = np.nan
        metrics["coverage_90"] = np.nan

    return metrics


__all__ = [
    "rmse",
    "negative_log_likelihood",
    "interval_coverage",
    "crps_gaussian",
    "compute_regression_metrics",
]
