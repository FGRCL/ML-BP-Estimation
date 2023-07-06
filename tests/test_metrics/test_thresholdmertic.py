from unittest import TestCase

from keras.metrics import MeanAbsoluteError
from numpy import arange, average, float32, mean

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
