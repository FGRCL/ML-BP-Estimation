import unittest

import numpy as np
import heartpy as hp
import tensorflow as tf

import src.preprocessing.transforms as transforms

class TestTransforms(unittest.TestCase):

    def test_extract_clean_windows(self):
        track, timer = hp.load_exampledata(2)
        sample_rate = hp.get_samplerate_datetime(timer, timeformat = '%Y-%m-%d %H:%M:%S.%f')

        windows = transforms.extract_clean_windows(track, sample_rate, 8, 2)

        self.assertEqual(102, len(windows))

    def test_scale_array(self):
        array = tf.constant([82.0, 49.0, 99.0, 35.0, 14.0])

        scaled = transforms.scale_array(array, None)

        self.assertAlmostEqual(0, tf.math.reduce_mean(scaled).numpy(), 3)
        self.assertAlmostEqual(1, tf.math.reduce_variance(scaled).numpy(), 3)

    def test_scale_array_large_array(self):
        array = tf.constant(np.arange(0.0, 10000.0, 1.0))

        scaled = transforms.scale_array(array, None)

        self.assertAlmostEqual(0, tf.math.reduce_mean(scaled).numpy(), 3)
        self.assertAlmostEqual(1, tf.math.reduce_variance(scaled).numpy(), 3)
