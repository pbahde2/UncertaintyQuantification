"""Fast regression tests for model APIs and mixture scoring."""

from __future__ import annotations

import inspect
import unittest

import numpy as np
import torch

from metrics import compute_regression_metrics
from models import DKLModel, DeepEnsembleModel, ExactGPModel, MCDropoutModel, SparseGPModel
from models._common import TORCH_DTYPE, GaussianMLP


class ModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4)
        cls.X = rng.normal(size=(10, 2)).astype(np.float32)
        cls.y = (cls.X[:, 0] - 0.4 * cls.X[:, 1]).astype(np.float32)
        cls.X_test = rng.normal(size=(3, 2)).astype(np.float32)

    def test_all_model_prediction_contracts(self):
        X_train, y_train = self.X[:8], self.y[:8]
        X_val, y_val = self.X[8:], self.y[8:]
        validation = {"X_val": X_val, "y_val": y_val}
        models = [
            ExactGPModel(X_train, y_train, max_epochs=1, **validation),
            SparseGPModel(X_train, y_train, n_inducing=4, max_epochs=1, **validation),
            DeepEnsembleModel(
                X_train, y_train, n_members=2, max_epochs=1, patience=1, **validation
            ),
            MCDropoutModel(
                X_train, y_train, n_samples=3, max_epochs=1, patience=1, **validation
            ),
            DKLModel(X_train, y_train, max_epochs=1, **validation),
        ]
        for model in models:
            mean, variance = model.predict(self.X_test)
            self.assertEqual(mean.shape, (3,))
            self.assertEqual(variance.shape, (3,))
            self.assertTrue(np.all(np.isfinite(mean)))
            self.assertTrue(np.all(np.isfinite(variance)))
            self.assertTrue(np.all(variance > 0.0))

    def test_all_methods_share_training_duration_defaults(self):
        for model_class in [
            ExactGPModel, SparseGPModel, DeepEnsembleModel, MCDropoutModel, DKLModel,
        ]:
            parameters = inspect.signature(model_class.__init__).parameters
            self.assertEqual(parameters["max_epochs"].default, 400)
            self.assertEqual(parameters["warmup_epochs"].default, 50)
            self.assertEqual(parameters["patience"].default, 5)

    def test_neural_models_use_two_equal_width_hidden_layers(self):
        network = GaussianMLP(input_dim=3)
        self.assertEqual(network.hidden1.in_features, 3)
        self.assertEqual(network.hidden1.out_features, 6)
        self.assertEqual(network.hidden2.in_features, 6)
        self.assertEqual(network.hidden2.out_features, 6)

        model = DKLModel(
            self.X[:8], self.y[:8], max_epochs=1,
            X_val=self.X[8:], y_val=self.y[8:],
        )
        linear_layers = [
            layer for layer in model.model.feature_extractor.network
            if isinstance(layer, torch.nn.Linear)
        ]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linear_layers],
            [(2, 4), (4, 4), (4, 2)],
        )
        self.assertEqual(model.feature_dim, self.X.shape[1])

    def test_all_models_use_shared_precision(self):
        validation = {"X_val": self.X[8:], "y_val": self.y[8:]}
        models = [
            ExactGPModel(self.X[:8], self.y[:8], max_epochs=1, **validation),
            SparseGPModel(
                self.X[:8], self.y[:8], n_inducing=4, max_epochs=1, **validation
            ),
            DeepEnsembleModel(
                self.X[:8], self.y[:8], n_members=2, max_epochs=1, **validation
            ),
            MCDropoutModel(
                self.X[:8], self.y[:8], n_samples=2, max_epochs=1, **validation
            ),
            DKLModel(self.X[:8], self.y[:8], max_epochs=1, **validation),
        ]
        for model in models:
            modules = model.members if hasattr(model, "members") else [model.model]
            for module in modules:
                self.assertEqual({p.dtype for p in module.parameters()}, {TORCH_DTYPE})

    def test_sparse_gp_keeps_selection_capacity_during_full_refit(self):
        model = SparseGPModel(
            self.X[:8], self.y[:8], n_inducing=256, max_epochs=1,
            X_val=self.X[8:], y_val=self.y[8:],
        )
        self.assertEqual(model.n_inducing_, 8)
        model.refit_full(self.X, self.y)
        self.assertEqual(model.n_inducing_, 8)
        self.assertEqual(model.n_inducing, 256)

    def test_all_methods_respect_validation_warmup(self):
        validation = {"X_val": self.X[8:], "y_val": self.y[8:]}
        models = [
            ExactGPModel(
                self.X[:8], self.y[:8], max_epochs=6, warmup_epochs=4,
                patience=1, **validation,
            ),
            SparseGPModel(
                self.X[:8], self.y[:8], n_inducing=4, max_epochs=6,
                warmup_epochs=4, patience=1, **validation,
            ),
            MCDropoutModel(
                self.X[:8], self.y[:8], n_samples=2, max_epochs=6,
                warmup_epochs=4, patience=1, **validation,
            ),
            DKLModel(
                self.X[:8], self.y[:8], max_epochs=6, warmup_epochs=4,
                patience=1, **validation,
            ),
        ]
        for model in models:
            self.assertGreaterEqual(model.best_epoch_, 4)

        ensemble = DeepEnsembleModel(
            self.X[:8], self.y[:8], n_members=2, max_epochs=6,
            warmup_epochs=4, patience=1, **validation,
        )
        self.assertTrue(all(epoch >= 4 for epoch in ensemble.selected_epochs_))


if __name__ == "__main__":
    unittest.main()
