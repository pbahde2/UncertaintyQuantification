"""Tests for the Concrete experiment's fixed-size data protocol."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments.concrete_experiment import (
    DATASET_NAME,
    TEST_SIZE,
    TRAIN_POOL_SIZE,
    concrete_dataset,
)


class ConcreteExperimentTests(unittest.TestCase):
    def setUp(self):
        # The first feature is a stable row identifier for checking the partition.
        self.X = np.column_stack(
            [np.arange(1_030, dtype=float), np.zeros((1_030, 7), dtype=float)]
        )
        self.y = np.arange(1_030, dtype=float)

    @patch("experiments.concrete_experiment.load_concrete_data")
    def test_requested_training_pool_and_test_sizes_are_disjoint(self, mocked_load):
        mocked_load.return_value = (self.X, self.y)

        data = concrete_dataset(DATASET_NAME, random_seed=4)

        self.assertEqual(len(data["X_train"]), TRAIN_POOL_SIZE)
        self.assertEqual(len(data["X_test"]), TEST_SIZE)
        train_ids = set(data["X_train"][:, 0])
        test_ids = set(data["X_test"][:, 0])
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertEqual(len(train_ids | test_ids), 1_030)

    @patch("experiments.concrete_experiment.load_concrete_data")
    def test_split_is_reproducible_for_a_seed(self, mocked_load):
        mocked_load.return_value = (self.X, self.y)

        first = concrete_dataset(DATASET_NAME, random_seed=7)
        second = concrete_dataset(DATASET_NAME, random_seed=7)

        np.testing.assert_array_equal(first["X_train"], second["X_train"])
        np.testing.assert_array_equal(first["X_test"], second["X_test"])


if __name__ == "__main__":
    unittest.main()
