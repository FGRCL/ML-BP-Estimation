from unittest import TestCase

from tensorflow import TensorShape

from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing
from tests.fixtures.dataset import DatasetLoaderFixture


class TestHeartbeatPreprocessing(TestCase):
    def test_can_preprocess(self):
        train, _, _ = DatasetLoaderFixture().load_datasets()
        pipeline = HeartbeatPreprocessing(
            125,
            5,
            (0.1, 8),
            30,
            230,
            400,
        )
        train = pipeline.apply(train)

        dataset = train.take(1)
        element = next(iter(dataset))

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(element)
        self.assertEqual((TensorShape([400, 1]), TensorShape([2])), dataset.output_shapes)
        self.assertEqual([400, 1], element[0].shape)
        self.assertEqual([2], element[1].shape)
