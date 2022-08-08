from unittest import TestCase

from neurokit2 import ppg_simulate
from tensorflow import TensorSpec, float64
from tensorflow.python.data import Dataset

from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


class TestWindowPreprocessing(TestCase):

    def test_output_shape(self):
        ppg_signal = (ppg_simulate(duration=120, sampling_rate=500) * 85) + 50
        dataset = Dataset.from_tensor_slices([ppg_signal])
        pipeline = WindowPreprocessing()

        processed_dataset = pipeline.preprocess(dataset)

        expected_specs = (TensorSpec(shape=4000, dtype=float64), TensorSpec(shape=2, dtype=float64))
        self.assertEqual(expected_specs, processed_dataset.element_spec)

    def test_has_data(self):
        ppg_signal = (ppg_simulate(duration=120, sampling_rate=500) * 85) + 50
        dataset = Dataset.from_tensor_slices([ppg_signal])
        pipeline = WindowPreprocessing()

        processed_dataset = pipeline.preprocess(dataset)

        try:
            element = next(iter(processed_dataset))
        except StopIteration:
            self.fail("Dataset has no elements")
        self.assertIsNotNone(element)
