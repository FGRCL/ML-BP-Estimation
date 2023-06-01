from unittest import TestCase

from mlbpestimation.data.saveddatasetloader import SavedDatasetLoader
from mlbpestimation.data.shufflesplitdatasetloader import ShuffleSplitDatasetLoader


class TestShuffleSplitDatasetLoader(TestCase):
    def test_mimic_window(self):
        original_train, original_val, original_test = SavedDatasetLoader('../../data/mimic-window').load_datasets()
        total = len(original_train) + len(original_val) + len(original_test)

        result_train, result_val, result_test = ShuffleSplitDatasetLoader(SavedDatasetLoader('../../data/mimic-window'), 0.7, 0.15, 0.15, 1337).load_datasets()

        self.assertAlmostEqual(len(result_train), int(total * 0.7), delta=100)
        self.assertAlmostEqual(len(result_val), int(total * 0.15), delta=100)
        self.assertAlmostEqual(len(result_test), int(total * 0.15), delta=100)
