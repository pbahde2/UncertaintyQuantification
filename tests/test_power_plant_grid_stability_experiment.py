"""Tests for the separate Power Plant/Electrical Grid experiment."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from execute_experiment import count_experiment_jobs
from experiments.power_plant_grid_stability_experiment import (
    GRID_DATASET_NAME,
    GRID_N_FEATURES,
    GRID_TEST_SIZE,
    GRID_TOTAL_OBSERVATIONS,
    GRID_TRAIN_POOL_SIZE,
    N_SPLITS,
    OUTPUT_PATH,
    TRAIN_SIZES,
    _extract_electrical_grid_arrays,
    build_model_factories,
    power_plant_grid_dataset,
)


class PowerPlantGridStabilityExperimentTests(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack(
            [
                np.arange(GRID_TOTAL_OBSERVATIONS, dtype=float),
                np.zeros(
                    (GRID_TOTAL_OBSERVATIONS, GRID_N_FEATURES - 1), dtype=float
                ),
            ]
        )
        self.y = np.arange(GRID_TOTAL_OBSERVATIONS, dtype=float)

    def test_extracts_continuous_target_and_removes_p1(self):
        features = pd.DataFrame(
            {
                "tau1": [1.0, 2.0],
                "p1": [3.0, 4.0],
                "g1": [5.0, 6.0],
            }
        )
        targets = pd.DataFrame(
            {"stab": [0.1, -0.2], "stabf": ["unstable", "stable"]}
        )
        dataset = SimpleNamespace(
            data=SimpleNamespace(features=features, targets=targets)
        )

        X, y = _extract_electrical_grid_arrays(dataset)

        np.testing.assert_array_equal(X, [[1.0, 5.0], [2.0, 6.0]])
        np.testing.assert_array_equal(y, [0.1, -0.2])

    @patch(
        "experiments.power_plant_grid_stability_experiment.load_electrical_grid_data"
    )
    def test_grid_split_sizes_and_disjointness(self, mocked_load):
        mocked_load.return_value = (self.X, self.y)

        data = power_plant_grid_dataset(GRID_DATASET_NAME, random_seed=4)

        self.assertEqual(len(data["X_train"]), GRID_TRAIN_POOL_SIZE)
        self.assertEqual(len(data["X_test"]), GRID_TEST_SIZE)
        train_ids = set(data["X_train"][:, 0])
        test_ids = set(data["X_test"][:, 0])
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertEqual(len(train_ids | test_ids), GRID_TOTAL_OBSERVATIONS)

    @patch(
        "experiments.power_plant_grid_stability_experiment.load_electrical_grid_data"
    )
    def test_grid_split_is_reproducible(self, mocked_load):
        mocked_load.return_value = (self.X, self.y)

        first = power_plant_grid_dataset(GRID_DATASET_NAME, random_seed=11)
        second = power_plant_grid_dataset(GRID_DATASET_NAME, random_seed=11)

        np.testing.assert_array_equal(first["X_train"], second["X_train"])
        np.testing.assert_array_equal(first["X_test"], second["X_test"])

    def test_new_run_has_distinct_output_and_expected_job_count(self):
        self.assertEqual(OUTPUT_PATH.name, "power_plant_grid_stability_experiment.csv")
        self.assertNotEqual(OUTPUT_PATH.name, "power_plant_experiment.csv")
        self.assertEqual(
            count_experiment_jobs(
                build_model_factories(), TRAIN_SIZES, n_datasets=2, n_splits=N_SPLITS
            ),
            1_520,
        )


if __name__ == "__main__":
    unittest.main()
