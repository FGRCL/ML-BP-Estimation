from unittest import TestCase

from mlbpestimation.preprocessing.pipelines.beatsequencepreprocessing import BeatSequencePreprocessing
from tests.fixtures.signaldatasetloaderfixture import SignalDatasetLoaderFixture
from tests.utils import get_dataset_output_shapes


class TestBeatSequencePreprocessing(TestCase):
    def test_can_preprocess(self):
        train, _, _ = SignalDatasetLoaderFixture().load_datasets()
        pipeline = BeatSequencePreprocessing(
            125,
            5,
            (0.1, 8),
            30,
            230,
            400,
            16,
            1,
            False,
            True,
        )
        train = pipeline.apply(train)

        dataset = train.take(1)
        element = next(iter(dataset))

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(element)
        self.assertEqual([[16, 400], [2]], get_dataset_output_shapes(dataset))
        self.assertEqual(element[0].shape, [16, 400])
        self.assertEqual(element[1].shape, [2])
