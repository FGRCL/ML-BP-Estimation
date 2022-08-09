from unittest import TestCase

from neurokit2 import ppg_simulate
from tensorflow import TensorSpec, float64
from tensorflow.python.data import Dataset

from src.data.mimic4.dataset import load_mimic_dataset
from src.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing
from src.vitaldb.casesplit import load_vitaldb_dataset


class TestWindowPreprocessing(TestCase):

    def test_output_shape(self):
        ppg_signal = (ppg_simulate(duration=120, sampling_rate=500) * 85) + 50
        dataset = Dataset.from_tensor_slices([ppg_signal])
        pipeline = WindowPreprocessing()

        processed_dataset = pipeline.preprocess(dataset)

        expected_specs = (TensorSpec(shape=(4000, 1), dtype=float64), TensorSpec(shape=2, dtype=float64))
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

    def test_preprocess_mimic(self):
        dataset, _, _ = load_mimic_dataset()
        dataset = dataset.take(1)
        pipeline = WindowPreprocessing(frequency=64)

        dataset = pipeline.preprocess(dataset)

        self.assertIsNotNone(dataset)

    def test_preprocess_vitaldb(self):
        dataset, _, _ = load_vitaldb_dataset()
        dataset = dataset.take(1)
        pipeline = WindowPreprocessing(frequency=500)

        dataset = pipeline.preprocess(dataset)

        self.assertIsNotNone(dataset)
