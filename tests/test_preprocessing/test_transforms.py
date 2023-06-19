import unittest

import heartpy as hp
import numpy as np
import tensorflow as tf

import mlbpestimation.preprocessing.shared.transforms as transforms


class TestTransforms(unittest.TestCase):

    def test_extract_clean_windows(self):
        track, timer = hp.load_exampledata(2)
        sample_rate = hp.get_samplerate_datetime(timer, timeformat='%Y-%m-%d %H:%M:%S.%f')

        windows = transforms.extract_clean_windows(track, sample_rate, 8, 2)

        self.assertEqual(102, len(windows))

    def test_scale_array(self):
        array = tf.constant([82.0, 49.0, 99.0, 35.0, 14.0])

        scaled = transforms.standardize_track(array, None)

        self.assertAlmostEqual(0, tf.math.reduce_mean(scaled).numpy(), 3)
        self.assertAlmostEqual(1, tf.math.reduce_variance(scaled).numpy(), 3)

    def test_scale_array_large_array(self):
        array = tf.constant(np.arange(0.0, 10000.0, 1.0))

        scaled = transforms.standardize_track(array, None)

        self.assertAlmostEqual(0, tf.math.reduce_mean(scaled).numpy(), 3)
        self.assertAlmostEqual(1, tf.math.reduce_variance(scaled).numpy(), 3)

    def test_filter_track(self):
        signal = np.zeros(10000)

        filteredTracks = transforms.filter_track(signal, 500)

        expectedTrack = np.zeros(10000)
        self.assertSequenceEqual(expectedTrack, filteredTracks.lowpass)
        self.assertSequenceEqual(expectedTrack, filteredTracks.bandpass)
