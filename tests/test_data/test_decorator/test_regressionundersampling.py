from unittest import TestCase

from numpy import empty, linspace, stack
from tensorflow.python.data import Dataset

from mlbpestimation.data.decorator.regressionundersampling import RegressionUndersampling
from tests.fixtures.datasetloaderfixture import DatasetLoaderFixture


class TestRegressionUndersampling(TestCase):
    def test_undersampling(self):
        input_signal = empty(1000)
        sbp = linspace(50, 230, 1000)
        dbp = linspace(30, 150, 1000)
        pressures = stack((sbp, dbp), axis=1)
        dataset = Dataset.from_tensor_slices((input_signal, pressures))
        dataset_loader = DatasetLoaderFixture(dataset, None, None)

        undersampled_dataset = RegressionUndersampling(dataset_loader, 50, 150, 0.5)
        resampled, _, _ = undersampled_dataset.load_datasets()

        minority_count = ((dbp <= 50) | (150 <= sbp)).sum()
        expected_total = min(1000, int(minority_count + minority_count / 0.5))
        total = resampled.reduce(0, lambda x, _: x + 1)
        self.assertIsNotNone(resampled)
        self.assertEqual(expected_total, total)
