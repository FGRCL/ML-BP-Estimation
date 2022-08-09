from unittest import TestCase

from src.data.mimic4.dataset import load_mimic_dataset


class TestMimicDataset(TestCase):

    def test_can_load_signal(self):
        dataset, _, _ = load_mimic_dataset()
        dataset = dataset.take(1)

        self.assertIsNotNone(dataset)
