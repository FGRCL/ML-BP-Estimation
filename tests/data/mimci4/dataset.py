from unittest import TestCase

from mlbpestimation.data.datasource.mimic4.dataset import MimicDataSource


class TestMimicDataset(TestCase):

    def test_can_load_signal(self):
        dataset, _, _ = MimicDataSource().get_datasets()
        dataset = dataset.take(1)

        self.assertIsNotNone(dataset)
