from unittest import TestCase

from tensorflow import TensorShape

from mlbpestimation.data.mimic4.mimicdatasetloader import MimicDatasetLoader
from mlbpestimation.preprocessing.pipelines.beatsequencepreprocessing import BeatSequencePreprocessing


class TestBeatSequencePreprocessing(TestCase):
    def test_can_preprocess(self):
        train, _, _ = MimicDatasetLoader('/Users/francois/git/ML-BP-Estimation/data/mimic-IV/physionet.org/files/mimic4wdb/0.1.0/waves', 1337,
                                         False).load_datasets()
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
