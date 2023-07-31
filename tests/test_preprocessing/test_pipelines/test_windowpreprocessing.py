from unittest import TestCase

from tensorflow import TensorShape

from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing
from tests.fixtures.signaldatasetloaderfixture import SignalDatasetLoaderFixture


class TestWindowPreprocessing(TestCase):
    def test_can_preprocess(self):
        train, _, _ = SignalDatasetLoaderFixture().load_datasets()
        pipeline = WindowPreprocessing(
            125,
            8,
            2,
            30,
            230,
            5,
            (0.1, 8),
            False,
        )

        dataset = pipeline.apply(train)
        element = next(iter(dataset))

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(element)
        self.assertEqual((TensorShape([1000, 1]), TensorShape([2])), dataset.output_shapes)
        self.assertEqual((1000, 1), element[0].shape)
        self.assertEqual((2,), element[1].shape)
