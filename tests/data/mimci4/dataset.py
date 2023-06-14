from unittest import TestCase

from mlbpestimation.data.mimic4.mimicdatasetloader import MimicDatasetLoader


class TestMimicDataset(TestCase):

    def test_can_load_signal(self):
        dataset, _, _ = MimicDatasetLoader('/Users/francois/git/ML-BP-Estimation/data/mimic-IV/physionet.org/files/mimic4wdb/0.1.0/waves', 106,
                                           use_ppg=False).load_datasets()

        element = next(iter(dataset))

        self.assertIsNotNone(element)
        self.assertEqual(len(element), 2)

    def test_load_with_ppg(self):
        dataset, _, _ = MimicDatasetLoader('/Users/francois/git/ML-BP-Estimation/data/mimic-IV/physionet.org/files/mimic4wdb/0.1.0/waves', 106,
                                           use_ppg=True).load_datasets()

        element = next(iter(dataset))

        self.assertIsNotNone(element)
        self.assertEqual(len(element), 2)
