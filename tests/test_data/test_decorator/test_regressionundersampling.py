from unittest import TestCase

from mlbpestimation.data.decorator.regressionundersampling import RegressionUndersampling
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestRegressionUndersampling(TestCase):
    def test_undersampling(self):
        window_dataset = WindowDatasetLoaderFixture()

        undersampled_dataset = RegressionUndersampling(window_dataset, 50, 150, 1.0)
        resampled, _, _ = undersampled_dataset.load_datasets()

        self.assertIsNotNone(resampled)
        # TODO needs better asserts
