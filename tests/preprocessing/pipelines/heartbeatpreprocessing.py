from unittest import TestCase

import tensorflow as tf
from neurokit2 import ppg_simulate

from src.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing


class TestHeartbeatPreprocessing(TestCase):

    def test_pipeline(self):
        ppg_signal = ppg_simulate(sampling_rate=500)
        dataset = tf.data.Dataset.from_tensor_slices([ppg_signal])
        pipeline = HeartbeatPreprocessing()

        processed_pipeline = pipeline.preprocess(dataset)

        print(processed_pipeline)
        for element in processed_pipeline:
            print(element)
