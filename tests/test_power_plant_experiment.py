"""Tests for the Combined Cycle Power Plant experiment protocol."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments.power_plant_experiment import (
    DATASET_NAME,
    PROTEIN_DATASET_NAME,
    PROTEIN_N_FEATURES,
    PROTEIN_TEST_SIZE,
    PROTEIN_TOTAL_OBSERVATIONS,
    PROTEIN_TRAIN_POOL_SIZE,
    TEST_SIZE,
    TRAIN_POOL_SIZE,
    TRAIN_SIZES,
    power_plant_dataset,
)


class PowerPlantExperimentTests(unittest.TestCase):
    def setUp(self):
        # The first feature is a stable row identifier for checking the partition.
        self.X = np.column_stack(
            [np.arange(9_568, dtype=float), np.zeros((9_568, 3), dtype=float)]
        )
        self.y = np.arange(9_568, dtype=float)

    @patch("experiments.power_plant_experiment.load_power_plant_data")
    def test_split_sizes_and_disjointness(self, mocked_load):
        mocked_load.return_value = (self.X, self.y)

        data = power_plant_dataset(DATASET_NAME, random_seed=4)

        self.assertEqual(len(data["X_train"]), TRAIN_POOL_SIZE)
        self.assertEqual(len(data["X_test"]), TEST_SIZE)
        self.assertLessEqual(max(TRAIN_SIZES), len(data["X_train"]))
        train_ids = set(data["X_train"][:, 0])
        test_ids = set(data["X_test"][:, 0])
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertEqual(len(train_ids | test_ids), 9_568)

    @patch("experiments.power_plant_experiment.load_power_plant_data")
    def test_split_is_reproducible_for_a_seed(self, mocked_load):
        mocked_load.return_value = (self.X, self.y)

        first = power_plant_dataset(DATASET_NAME, random_seed=11)
        second = power_plant_dataset(DATASET_NAME, random_seed=11)

        np.testing.assert_array_equal(first["X_train"], second["X_train"])
        np.testing.assert_array_equal(first["X_test"], second["X_test"])

    @patch("experiments.power_plant_experiment.load_protein_structure_data")
    def test_protein_split_sizes_and_disjointness(self, mocked_load):
        X = np.column_stack(
            [
                np.arange(PROTEIN_TOTAL_OBSERVATIONS, dtype=float),
                np.zeros(
                    (PROTEIN_TOTAL_OBSERVATIONS, PROTEIN_N_FEATURES - 1),
                    dtype=float,
                ),
            ]
        )
        y = np.arange(PROTEIN_TOTAL_OBSERVATIONS, dtype=float)
        mocked_load.return_value = (X, y)

        data = power_plant_dataset(PROTEIN_DATASET_NAME, random_seed=4)

        self.assertEqual(len(data["X_train"]), PROTEIN_TRAIN_POOL_SIZE)
        self.assertEqual(len(data["X_test"]), PROTEIN_TEST_SIZE)
        train_ids = set(data["X_train"][:, 0])
        test_ids = set(data["X_test"][:, 0])
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertEqual(len(train_ids | test_ids), PROTEIN_TOTAL_OBSERVATIONS)


if __name__ == "__main__":
    unittest.main()
