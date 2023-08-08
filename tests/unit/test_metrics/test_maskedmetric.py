from unittest import TestCase

from keras.metrics import MeanAbsoluteError
from numpy import array, mean

from mlbpestimation.metrics.maskedmetric import MaskedMetric


class TestMaskedMetric(TestCase):
    def test_single_batch(self):
        metric = MaskedMetric(MeanAbsoluteError(), [True, False])
        y_true = array([[1, 2, 3, 4], [4, 5, 6, 7]])
        y_pred = array([[14, 15, 16, 17], [18, 19, 20, 21]])

        metric.update_state(y_true.T, y_pred.T)
        result = metric.result()

        expected = mean(abs(y_true[0] - y_pred[0]))
        self.assertEqual(expected, result)

    def test_single_batch_second_axis(self):
        metric = MaskedMetric(MeanAbsoluteError(), [False, True])
        y_true = array([[1, 2, 3, 4], [4, 5, 6, 7]])
        y_pred = array([[14, 15, 16, 17], [18, 19, 20, 21]])

        metric.update_state(y_true.T, y_pred.T)
        result = metric.result()

        expected = mean(abs(y_true[1] - y_pred[1]))
        self.assertEqual(expected, result)

    def test_multiple_batches(self):
        metric = MaskedMetric(MeanAbsoluteError(), [True, False])
        y_true_batches = array([
            [[44, 23, 37, 39], [6, 23, 48, 18]],
            [[35, 1, 50, 21], [25, 18, 12, 3]],
            [[31, 31, 10, 39], [15, 7, 5, 35]]
        ])
        y_pred_batches = array([
            [[30, 8, 44, 26], [0, 40, 15, 50]],
            [[33, 45, 46, 43], [37, 3, 31, 27]],
            [[37, 20, 42, 13], [24, 1, 2, 10]]
        ])

        for y_true, y_pred in zip(y_true_batches, y_pred_batches):
            metric.update_state(y_true.T, y_pred.T)
        result = metric.result()

        expected = mean(abs(y_true_batches[:, 0].ravel() - y_pred_batches[:, 0].ravel()))
        self.assertEqual(expected, result)
