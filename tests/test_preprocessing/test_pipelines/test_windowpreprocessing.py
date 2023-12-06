from unittest import TestCase

from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing
from tests.fixtures.signaldatasetloaderfixture import SignalDatasetLoaderFixture
from tests.utils import get_dataset_output_shapes


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
            True,
        )

        dataset = pipeline.apply(train)
        element = next(iter(dataset))

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(element)
        self.assertEqual([[1000, 1], [2]], get_dataset_output_shapes(dataset))
        self.assertEqual((1000, 1), element[0].shape)
        self.assertEqual((2,), element[1].shape)
