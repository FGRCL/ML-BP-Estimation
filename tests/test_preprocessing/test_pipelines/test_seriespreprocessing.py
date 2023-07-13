from unittest import TestCase

from tensorflow import TensorShape

from mlbpestimation.preprocessing.pipelines.seriespreprocessing import SeriesPreprocessing
from tests.fixtures.signaldatasetloaderfixture import SignalDatasetLoaderFixture


class TestSeriesPreprocessing(TestCase):
    def test_can_preprocess(self):
        train, _, _ = SignalDatasetLoaderFixture(720).load_datasets()
        pipeline = SeriesPreprocessing(
            125,
            0.2,
            5,
            (0.1, 8),
        )

        dataset = pipeline.apply(train)
        element = next(iter(dataset))

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(element)
        self.assertEqual((TensorShape([None, 1]), TensorShape([None, 1])), dataset.output_shapes)
        self.assertEqual((90000, 1), element[0].shape)
        self.assertEqual((90000, 1), element[1].shape)
