from unittest import TestCase

from numpy import array, mean

from mlbpestimation.metrics.meanprediction import MeanPrediction


class TestMeanPrediction(TestCase):
    def test_single_batch(self):
        metric = MeanPrediction()
        y_true = array([[78, 39, 40, 94, 21, 35, 90, 88, 14, 55]])
        y_pred = array([[98, 24, 29, 16, 61, 62, 77, 52, 36, 83]])

        metric.update_state(y_true.T, y_pred.T)
        result = metric.result()

        expected = mean(y_pred)
        self.assertEqual(expected, result)

    def test_multiple_batches(self):
        metric = MeanPrediction()
        y_true_batches = array([
            [[15, 73, 87, 37, 46]],
            [[39, 13, 68, 43, 4]],
            [[94, 7, 48, 29, 70]],
            [[46, 61, 70, 41, 14]],
            [[60, 12, 50, 78, 28]]
        ])
        y_pred_batches = array([
            [[83, 84, 97, 95, 43]],
            [[81, 100, 56, 30, 87]],
            [[29, 98, 38, 45, 82]],
            [[78, 50, 44, 19, 52]],
            [[49, 39, 83, 48, 7]]
        ])

        for y_true, y_pred in zip(y_true_batches, y_pred_batches):
            metric.update_state(y_true.T, y_pred.T)
            result = metric.result()

        expected = mean(y_pred_batches)
        self.assertEqual(expected, result)
