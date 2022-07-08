from unittest import TestCase

import tensorflow as tf
from neurokit2 import ppg_simulate
from tensorflow import TensorSpec, float64

from src.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing


class TestHeartbeatPreprocessing(TestCase):

    def test_pipeline(self):
        ppg_signal = (ppg_simulate(duration=3600, sampling_rate=500) * 100) + 30
        dataset = tf.data.Dataset.from_tensor_slices([ppg_signal])
        pipeline = HeartbeatPreprocessing()

        processed_dataset = pipeline.preprocess(dataset)

        expected_specs = (TensorSpec(shape=400, dtype=float64), TensorSpec(shape=2, dtype=float64))
        self.assertEqual(expected_specs, processed_dataset.element_spec)
