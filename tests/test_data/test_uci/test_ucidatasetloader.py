from unittest import TestCase
from unittest.mock import MagicMock, patch

from numpy import ones

from mlbpestimation.data.uci.ucidatasetloader import UciDatasetLoader
from tests.data.directories import uci


class TestUciDatasetLoader(TestCase):
    @patch('mlbpestimation.data.uci.ucidatasetloader.loadmat')
    def test_can_load_signal(self, mock_loadmat: MagicMock):
        mock_loadmat.return_value = {
            'a': ones((200, 2, 1000)),
            'b': ones((100, 2, 500)),
            'c': ones((50, 2, 5000))
        }
        dataset, _, _ = UciDatasetLoader(uci, 125, 1337, use_ppg=False).load_datasets()

        element = next(iter(dataset))

        self.assertIsNotNone(element)
        self.assertEqual(len(element), 2)

    @patch('mlbpestimation.data.uci.ucidatasetloader.loadmat')
    def test_load_with_ppg(self, mock_loadmat: MagicMock):
        mock_loadmat.return_value = {
            'a': ones((200, 2, 1000)),
            'b': ones((100, 2, 500)),
            'c': ones((50, 2, 5000))
        }
        dataset, _, _ = UciDatasetLoader(uci, 125, 1337, use_ppg=True).load_datasets()

        element = next(iter(dataset))

        self.assertIsNotNone(element)
        self.assertEqual(len(element), 2)
