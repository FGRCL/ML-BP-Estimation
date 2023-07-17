from unittest import TestCase

from keras.metrics import MeanAbsoluteError
from numpy import arange, asarray, average, float32, mean

from mlbpestimation.metrics.thresholdmetric import ThresholdMetric


class TestThresholdMetric(TestCase):
    def test_lower(self):
        metric = ThresholdMetric(MeanAbsoluteError(), lower=10)
        y_true = arange(0, 20, dtype=float32)
        y_pred = arange(20, 0, -1, dtype=float32)

        metric.update_state(y_true, y_pred)
        result = metric.result()

        expected = mean(abs(y_true[11:] - y_pred[11:]))
        self.assertEqual(expected, result)

    def test_upper(self):
        metric = ThresholdMetric(MeanAbsoluteError(), upper=10)
        y_true = arange(0, 20, dtype=float32)
        y_pred = arange(20, 0, -1, dtype=float32)

        metric.update_state(y_true, y_pred)
        result = metric.result()

        expected = mean(abs(y_true[:10] - y_pred[:10]))
        self.assertEqual(expected, result)

    def test_sample_weight(self):
        metric = ThresholdMetric(MeanAbsoluteError(), lower=5, upper=15)
        y_true = arange(0, 20, dtype=float32)
        y_pred = arange(20, 0, -1, dtype=float32)
        sample_weight = arange(0, 20, dtype=float32)

        metric.update_state(y_true, y_pred)
        result = metric.result()

        expected = average(abs(y_true[6:15] - y_pred[6:15]), weights=sample_weight[6:15])
        self.assertEqual(expected, result)

    def test_upper_lower_single_batch(self):
        metric = ThresholdMetric(MeanAbsoluteError(), lower=5, upper=15)
        y_true = arange(0, 20, dtype=float32)
        y_pred = arange(20, 0, -1, dtype=float32)

        metric.update_state(y_true, y_pred)
        result = metric.result()

        expected = mean(abs(y_true[6:15] - y_pred[6:15]))
        self.assertEqual(expected, result)

    def test_upper_lower_multiple_batch(self):
        metric = ThresholdMetric(MeanAbsoluteError(), lower=5, upper=15)
        y_true_batches = arange(0, 20, dtype=float32).repeat(5).reshape(20, 5).T
        y_pred_batches = arange(20, 0, -1, dtype=float32).repeat(5).reshape(20, 5).T

        for y_true, y_pred in zip(y_true_batches, y_pred_batches):
            metric.update_state(y_true, y_pred)
        result = metric.result()

        expected = mean(abs(y_true_batches[:, 6:15] - y_pred_batches[:, 6:15]))
        self.assertEqual(expected, result)

    def test_empty_array(self):
        metric = ThresholdMetric(MeanAbsoluteError(), lower=5, upper=15)
        y_true = asarray([])
        y_pred = asarray([])

        metric.update_state(y_true, y_pred)
        result = metric.result()

        expected = 0.0
        self.assertEqual(expected, result)

    def test_multiple_batches_with_empty_array(self):
        metric = ThresholdMetric(MeanAbsoluteError(), lower=0, upper=100)
        y_true_batches = ([
            asarray([39, 22, 73]),
            asarray([]),
            asarray([22, 50, 23])
        ])
        y_pred_batches = ([
            asarray([80, 59, 52]),
            asarray([]),
            asarray([87, 8, 38]),
        ])

        for y_true, y_pred in zip(y_true_batches, y_pred_batches):
            metric.update_state(y_true, y_pred)
        result = metric.result()

        absolute_errors = []
        for y_true, y_pred in zip(y_true_batches, y_pred_batches):
            absolute_errors += list(abs(y_true - y_pred))
        expected = mean(absolute_errors)
        self.assertAlmostEqual(expected, result.numpy(), 5)
