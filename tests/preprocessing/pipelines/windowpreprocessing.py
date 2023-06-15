from unittest import TestCase

from neurokit2 import ppg_simulate
from tensorflow import TensorShape, TensorSpec, float32
from tensorflow.python.data import Dataset

from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing
from tests.fixtures.dataset import DatasetLoaderFixture


class TestWindowPreprocessing(TestCase):
    def test_output_shape(self):
        ppg_signal = (ppg_simulate(duration=120, sampling_rate=500) * 85) + 50
        dataset = Dataset.from_tensor_slices([ppg_signal])
        pipeline = WindowPreprocessing()

        processed_dataset = pipeline.apply(dataset)

        expected_specs = (TensorSpec(shape=(4000, 1), dtype=float32), TensorSpec(shape=2, dtype=float32))
        self.assertEqual(expected_specs, processed_dataset.element_spec)

    def test_has_data(self):
        ppg_signal = (ppg_simulate(duration=120, sampling_rate=500) * 85) + 50
        dataset = Dataset.from_tensor_slices([ppg_signal])
        pipeline = WindowPreprocessing()

        processed_dataset = pipeline.apply(dataset)

        try:
            element = next(iter(processed_dataset))
        except StopIteration:
            self.fail("Dataset has no elements")
        self.assertIsNotNone(element)

    def test_preprocess_mimic(self):
        train, _, _ = DatasetLoaderFixture().load_datasets()
        pipeline = WindowPreprocessing(
            125,
            8,
            2,
            30,
            230,
            5,
            (0.1, 8),
        )

        dataset = pipeline.apply(train)
        element = next(iter(dataset))

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(element)
        self.assertEqual((TensorShape([1000, 1]), TensorShape([2])), dataset.output_shapes)
        self.assertEqual((1000, 1), element[0].shape)
        self.assertEqual((2,), element[1].shape)
