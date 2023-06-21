from unittest import TestCase

from mlbpestimation.data.vitaldb.vitaldatasetloader import VitalDatasetLoader


class TestVitalDatasetLoader(TestCase):
    def test_can_load_signal(self):
        dataset, _, _ = VitalDatasetLoader(1337, use_ppg=False).load_datasets()

        element = next(iter(dataset))

        self.assertIsNotNone(element)
        self.assertEqual(len(element), 2)

    def test_load_with_ppg(self):
        dataset, _, _ = VitalDatasetLoader(1337, use_ppg=True).load_datasets()

        element = next(iter(dataset))

        self.assertIsNotNone(element)
        self.assertEqual(len(element), 2)
