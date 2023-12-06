from unittest import TestCase

from mlbpestimation.preprocessing.pipelines.seriespreprocessing import SeriesPreprocessing
from tests.fixtures.signaldatasetloaderfixture import SignalDatasetLoaderFixture
from tests.utils import get_dataset_output_shapes


class TestSeriesPreprocessing(TestCase):
    def test_can_preprocess(self):
        train, _, _ = SignalDatasetLoaderFixture(720).load_datasets()
        pipeline = SeriesPreprocessing(
            125,
            20,
            10,
            5,
            (0.1, 8),
            False,
            True,
        )

        dataset = pipeline.apply(train)
        element = next(iter(dataset))

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(element)
        self.assertEqual([[None, 1], [None, 1]], get_dataset_output_shapes(dataset))
        self.assertEqual((12000, 1), element[0].shape)
        self.assertEqual((12000, 1), element[1].shape)
