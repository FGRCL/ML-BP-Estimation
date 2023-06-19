from unittest import TestCase

from keras.losses import MeanAbsoluteError
from numpy import arange

from mlbpestimation.losses.totalmeanabsoluteerror import TotalMeanAbsoluteErrorLoss


class TestTotalMeanAbsoluteErrorLoss(TestCase):
    def test_happy_path(self):
        pred = arange(1, 10).reshape((3, 3))
        true = arange(11, 20).reshape((3, 3))

        loss = TotalMeanAbsoluteErrorLoss()(pred, true)

        self.assertEqual(30, loss)

    def test_mae_comparisonm(self):
        pred = arange(1, 10).reshape((3, 3))
        true = arange(11, 29, 2).reshape((3, 3))

        tmae_loss = TotalMeanAbsoluteErrorLoss()(pred, true)
        mae_loss = MeanAbsoluteError()(pred, true)

        self.assertEqual(42, tmae_loss)
        self.assertEqual(14, mae_loss)
