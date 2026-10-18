"""Tests for leakage-free, nested benchmark preparation."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from execute_experiment import ModelSpec, count_experiment_jobs, execute_experiment
from data_preparation import nested_training_subsets, split_train_validation


class ProtocolTests(unittest.TestCase):
    def test_sparse_models_run_only_above_their_inducing_point_count(self):
        trained_sizes = []

        class Recorder:
            def __init__(self, X_train, y_train, **kwargs):
                trained_sizes.append(len(X_train) + len(kwargs["X_val"]))

            def predict(self, X_test):
                return np.zeros(len(X_test)), np.ones(len(X_test))

        factories = {
            "baseline": lambda X, y, **kwargs: Recorder(X, y, **kwargs),
            "sparse_gp_m5": ModelSpec(
                lambda X, y, **kwargs: Recorder(X, y, **kwargs), n_inducing=5
            ),
        }
        result = execute_experiment(
            factories,
            ["toy"],
            [4, 5, 6],
            n_splits=1,
            validation_fraction=0.25,
            dataset_loader=lambda name, random_seed=0: {
                "X_train": np.arange(8, dtype=float).reshape(-1, 1),
                "y_train": np.arange(8, dtype=float),
                "X_test": np.array([[8.0]]),
                "y_test": np.array([8.0]),
            },
        )

        sparse_rows = result[result["model"] == "sparse_gp_m5"]
        self.assertEqual(sparse_rows["train_size"].tolist(), [6])
        self.assertEqual(count_experiment_jobs(factories, [4, 5, 6]), 4)
        self.assertEqual(trained_sizes, [4, 5, 6, 6])

    def test_subsets_and_validation_roles_remain_nested(self):
        X = np.arange(30, dtype=float).reshape(-1, 1)
        y = np.arange(30)
        subsets = nested_training_subsets(X, y, [10, 20], random_seed=7)
        small_fit, small_val, small_y_fit, small_y_val = split_train_validation(*subsets[10])
        large_fit, large_val, large_y_fit, large_y_val = split_train_validation(*subsets[20])

        self.assertTrue(set(subsets[10][1]).issubset(set(subsets[20][1])))
        self.assertTrue(set(small_y_fit).issubset(set(large_y_fit)))
        self.assertTrue(set(small_y_val).issubset(set(large_y_val)))
        self.assertEqual(len(small_fit) + len(small_val), 10)

    def test_benchmark_scales_from_fit_partition_only(self):
        captured = {}

        def loader(_name, random_seed=0):
            return {
                "X_train": np.arange(20, dtype=float).reshape(-1, 1),
                "y_train": np.arange(20, dtype=float),
                "X_test": np.array([[100.0], [110.0]]),
                "y_test": np.array([0.0, 0.0]),
            }

        class Recorder:
            def __init__(self, X_train, y_train, **kwargs):
                captured["X_train"] = X_train.copy()
                captured["y_train"] = y_train.copy()
                captured.update(kwargs)

            def predict(self, X_test):
                captured["X_test"] = X_test.copy()
                return np.zeros(len(X_test)), np.ones(len(X_test))

        execute_experiment(
            {"recorder": lambda X, y, **kwargs: Recorder(X, y, **kwargs)},
            ["toy"], [10], n_splits=1, validation_fraction=0.2,
            dataset_loader=loader,
        )

        raw_subset = nested_training_subsets(
            np.arange(20, dtype=float).reshape(-1, 1), np.arange(20), [10], random_seed=0
        )[10]
        X_fit, X_val, y_fit, y_val = split_train_validation(*raw_subset)
        scaler = StandardScaler().fit(X_fit)
        np.testing.assert_allclose(captured["X_train"], scaler.transform(X_fit))
        np.testing.assert_allclose(captured["X_val"], scaler.transform(X_val))
        np.testing.assert_allclose(captured["X_test"], scaler.transform([[100.0], [110.0]]))
        np.testing.assert_array_equal(captured["y_train"], y_fit)
        np.testing.assert_array_equal(captured["y_val"], y_val)
        self.assertEqual(captured["random_seed"], 0)

    def test_predict_variance_is_converted_to_standard_deviation_for_metrics(self):
        class VarianceModel:
            def __init__(self, X_train, y_train, **kwargs):
                self.X_train = X_train
                self.y_train = y_train

            def predict(self, X_test):
                return np.array([0.0, 1.0]), np.array([4.0, 9.0])

        result = execute_experiment(
            {"variance_model": lambda X_train, y_train, **kwargs: VarianceModel(X_train, y_train, **kwargs)},
            ["toy"], [2], n_splits=1, validation_fraction=0.1,
            dataset_loader=lambda name, random_seed=0: {
                "X_train": np.array([[0.0], [1.0], [2.0], [3.0]]),
                "y_train": np.array([0.0, 1.0, 2.0, 3.0]),
                "X_test": np.array([[4.0], [5.0]]),
                "y_test": np.array([4.0, 5.0]),
            },
        )

        self.assertEqual(len(result), 1)
        self.assertTrue(np.isfinite(result["nll"]).all())
        self.assertTrue(np.isfinite(result["crps"]).all())
        self.assertTrue(np.isfinite(result["coverage_90"]).all())

    def test_selected_epoch_is_used_to_refit_on_the_complete_subset(self):
        captured = {}

        class RefitRecorder:
            def __init__(self, X_train, y_train, **kwargs):
                captured["selection_size"] = len(X_train)
                self.best_epoch_ = 3

            def refit_full(self, X_train, y_train):
                captured["refit_size"] = len(X_train)
                captured["refit_mean"] = np.mean(X_train, axis=0)
                return self

            def predict(self, X_test):
                captured["final_test"] = X_test.copy()
                return np.zeros(len(X_test)), np.ones(len(X_test))

        raw_train = np.arange(20, dtype=float).reshape(-1, 1)
        raw_test = np.array([[100.0], [110.0]])
        result = execute_experiment(
            {"recorder": lambda X, y, **kwargs: RefitRecorder(X, y, **kwargs)},
            ["toy"], [10], n_splits=1, validation_fraction=0.2,
            dataset_loader=lambda name, random_seed=0: {
                "X_train": raw_train,
                "y_train": np.arange(20, dtype=float),
                "X_test": raw_test,
                "y_test": np.zeros(2),
            },
        )

        X_sub, _ = nested_training_subsets(raw_train, np.arange(20), [10], 0)[10]
        full_scaler = StandardScaler().fit(X_sub)
        self.assertEqual(captured["selection_size"], 8)
        self.assertEqual(captured["refit_size"], 10)
        np.testing.assert_allclose(captured["refit_mean"], 0.0, atol=1e-12)
        np.testing.assert_allclose(captured["final_test"], full_scaler.transform(raw_test))
        self.assertEqual(result.loc[0, "selected_epoch"], 3.0)
        self.assertGreaterEqual(result.loc[0, "final_training_time_seconds"], 0.0)
        self.assertGreaterEqual(result.loc[0, "inference_time_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
