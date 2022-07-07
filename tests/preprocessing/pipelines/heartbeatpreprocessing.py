from unittest import TestCase

import matplotlib.pyplot
import tensorflow as tf
from neurokit2 import ppg_simulate

from src.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing


class TestHeartbeatPreprocessing(TestCase):

    def test_pipeline(self):
        ppg_signal = (ppg_simulate(duration=3600, sampling_rate=500) * 100) + 30
        dataset = tf.data.Dataset.from_tensor_slices([ppg_signal])
        pipeline = HeartbeatPreprocessing()

        processed_pipeline = pipeline.preprocess(dataset)

        for element in processed_pipeline:
            print(element[1].numpy())
            matplotlib.pyplot.plot(element[0].numpy())

        matplotlib.pyplot.show()
