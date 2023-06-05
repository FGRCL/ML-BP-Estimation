from unittest import TestCase

from mlbpestimation.data.mimic4.mimicdatasetloader import MimicDatasetLoader
from mlbpestimation.preprocessing.pipelines.beatsequencepreprocessing import BeatSequencePreprocessing
from tests.constants import data_directory


class TestBeatSequencePreprocessing(TestCase):
    def test_can_preprocess(self):
        train, _, _ = MimicDatasetLoader(data_directory / 'mimic-IV/physionet.org/files/mimic4wdb/0.1.0/waves', 1337).load_datasets()
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
        result = train.take(1)

        self.assertIsNotNone(result)
        self.assertEqual(result.output_shapes, [(16, 400, 1), (16, 2)])
