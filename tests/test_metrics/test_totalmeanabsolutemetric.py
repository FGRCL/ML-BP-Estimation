from unittest import TestCase

from numpy import array, mean, transpose

from mlbpestimation.metrics.totalmeanabsoluteerror import TotalMeanAbsoluteErrorMetric


class TestTotalMeanAbsoluteMetric(TestCase):
    def test_single_batch(self):
        metric = TotalMeanAbsoluteErrorMetric()
        y_true = array([[1, 2, 3, 4], [4, 5, 6, 7]])
        y_pred = array([[14, 15, 16, 17], [18, 19, 20, 21]])

        metric.update_state(y_true.T, y_pred.T)
        result = metric.result()

        expected = sum([mean(abs(true - pred)) for true, pred in zip(y_true, y_pred)])
        self.assertEqual(expected, result)

    def test_multiple_batches(self):
        metric = TotalMeanAbsoluteErrorMetric()
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

        means = []
        for y_true, y_pred in zip(transpose(y_true_batches, (1, 0, 2)), transpose(y_pred_batches, (1, 0, 2))):
            means.append(mean(abs(y_true.ravel() - y_pred.ravel())))
        expected = sum(means)
        self.assertEqual(expected, result)
