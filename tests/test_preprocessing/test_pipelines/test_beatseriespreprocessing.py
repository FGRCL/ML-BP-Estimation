from unittest import TestCase

from numpy import argmax
from scipy.signal import correlate, correlation_lags
from tensorflow import float32

from mlbpestimation.preprocessing.shared.transforms import AdjustPhaseLag
from tests.fixtures.signaldatasetloaderfixture import SignalDatasetLoaderFixture


class TestBeatSeriesPreprocessing(TestCase):
    def test_adjust_phase_lag(self):
        train, _, _ = SignalDatasetLoaderFixture().load_datasets()
        input_signal, output_signal = next(iter(train))
        transformer = AdjustPhaseLag((float32))

        input_result, output_result = transformer.transform(input_signal, output_signal)

        mode = 'full'
        correlation = correlate(input_result, output_result, mode=mode)
        lags = correlation_lags(input_result.shape[0], output_result.shape[0], mode=mode)
        phase_lag = lags[argmax(correlation)]

        self.assertEqual(phase_lag, 0)
