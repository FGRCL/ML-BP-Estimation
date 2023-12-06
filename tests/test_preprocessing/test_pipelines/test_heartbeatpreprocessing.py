from unittest import TestCase

from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing
from tests.fixtures.signaldatasetloaderfixture import SignalDatasetLoaderFixture
from tests.utils import get_dataset_output_shapes


class TestHeartbeatPreprocessing(TestCase):
    def test_can_preprocess(self):
        train, _, _ = SignalDatasetLoaderFixture().load_datasets()
        pipeline = HeartbeatPreprocessing(
            125,
            5,
            (0.1, 8),
            30,
            230,
            400,
            False,
            True,
        )
        train = pipeline.apply(train)

        dataset = train.take(1)
        element = next(iter(dataset))

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(element)
        self.assertEqual([[400, 1], [2]], get_dataset_output_shapes(dataset))
        self.assertEqual([400, 1], element[0].shape)
        self.assertEqual([2], element[1].shape)
