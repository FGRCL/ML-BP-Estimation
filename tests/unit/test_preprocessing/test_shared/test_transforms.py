from unittest import TestCase

import tensorflow as tf
from numpy import mean, std
from tensorflow import float64, ones, reshape

from mlbpestimation.preprocessing.shared.transforms import StandardScaling


class TestTransforms(TestCase):
    def test_standard_scaling_per_input_2d(self):
        data = reshape(tf.range(0, 1000000, dtype=float64), (-1, 1000))
        pressures = ones((data.shape[0], 2))
        scaler = StandardScaling(axis=1)

        scaled, pressures = scaler.transform(data, pressures)

        self.assertIsNotNone(scaled)
        for window in scaled:
            mu = mean(window)
            variance = std(window)
            self.assertAlmostEqual(mu, 0)
            self.assertAlmostEqual(variance, 1)

    def test_standard_scaling_per_signal_2d(self):
        data = reshape(tf.range(0, 1000000, dtype=float64), (-1, 1000))
        pressures = ones((data.shape[0], 2))
        scaler = StandardScaling(axis=-1)

        scaled, pressures = scaler.transform(data, pressures)

        self.assertIsNotNone(scaled)
        mu = mean(scaled)
        variance = std(scaled)
        self.assertAlmostEqual(mu, 0)
        self.assertAlmostEqual(variance, 1)

    def test_standard_scaling_per_input_3d(self):
        data = reshape(tf.range(0, 1000000, dtype=float64), (-1, 10, 100))
        pressures = ones((data.shape[0], 2))
        scaler = StandardScaling(axis=2)

        scaled, pressures = scaler.transform(data, pressures)

        self.assertIsNotNone(scaled)
        for sequence in scaled:
            for window in sequence:
                mu = mean(window)
                variance = std(window)
                self.assertAlmostEqual(mu, 0)
                self.assertAlmostEqual(variance, 1)

    def test_standard_scaling_per_signal_3d(self):
        data = reshape(tf.range(0, 1000000, dtype=float64), (-1, 10, 100))
        pressures = ones((data.shape[0], 2))
        scaler = StandardScaling(axis=-1)

        scaled, pressures = scaler.transform(data, pressures)

        self.assertIsNotNone(scaled)
        mu = mean(scaled)
        variance = std(scaled)
        self.assertAlmostEqual(mu, 0)
        self.assertAlmostEqual(variance, 1)
