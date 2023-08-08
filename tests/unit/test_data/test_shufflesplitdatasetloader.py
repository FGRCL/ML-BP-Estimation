from unittest import TestCase

from tests.unit.fixtures.signaldatasetloaderfixture import SignalDatasetLoaderFixture


class TestShuffleSplitDatasetLoader(TestCase):
    def test_mimic_window(self):
        original_train, original_val, original_test = SignalDatasetLoaderFixture().load_datasets()
        total = len(original_train) + len(original_val) + len(original_test)

        result_train, result_val, result_test = SignalDatasetLoaderFixture().load_datasets()

        self.assertAlmostEqual(len(result_train), int(total * 0.7), delta=100)
        self.assertAlmostEqual(len(result_val), int(total * 0.15), delta=100)
        self.assertAlmostEqual(len(result_test), int(total * 0.15), delta=100)
