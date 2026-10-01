"""Model implementations for the uncertainty benchmark.

Each file under this package contains one model class or factory function. The shared
interface is intentionally simple: train on (X_train, y_train), then return predictive
mean and predictive standard deviation on a new set of inputs X_test.
"""

from .exact_gp import ExactGPModel
from .sparse_gp import SparseGPModel
from .deep_ensemble import DeepEnsembleModel
from .mc_dropout import MCDropoutModel
from .dkl import DKLModel

__all__ = [
    "ExactGPModel",
    "SparseGPModel",
    "DeepEnsembleModel",
    "MCDropoutModel",
    "DKLModel",
]
