from unittest import TestCase

from mlbpestimation.data.mimic4 import MimicDatabase


class TestMimicDataset(TestCase):

    def test_can_load_signal(self):
        dataset, _, _ = MimicDatabase().load_datasets()
        dataset = dataset.take(1)

        self.assertIsNotNone(dataset)
