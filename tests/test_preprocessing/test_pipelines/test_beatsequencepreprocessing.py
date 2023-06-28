from unittest import TestCase

from tensorflow import TensorShape

from mlbpestimation.preprocessing.pipelines.beatsequencepreprocessing import BeatSequencePreprocessing
from tests.fixtures.signaldatasetloaderfixture import SignalDatasetLoaderFixture


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
            1
        )
        train = pipeline.apply(train)

        dataset = train.take(1)
        element = next(iter(dataset))

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(element)
        self.assertEqual((TensorShape([16, 400, 1]), TensorShape([2])), dataset.output_shapes)
        self.assertEqual(element[0].shape, [16, 400, 1])
        self.assertEqual(element[1].shape, [2])
