from unittest import TestCase

from mlbpestimation.data.datasource.mimic4.mimicdatabase import MimicDatabase


class TestMimicDataset(TestCase):

    def test_can_load_signal(self):
        dataset, _, _ = MimicDatabase().get_datasets()
        dataset = dataset.take(1)

        self.assertIsNotNone(dataset)
