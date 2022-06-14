import unittest

import numpy as np
import heartpy as hp

import src.preprocessing.transforms as transforms

class TestTransforms(unittest.TestCase):

    def test_extract_clean_windows(self):
        track, timer = hp.load_exampledata(2)
        sample_rate = hp.get_samplerate_datetime(timer, timeformat = '%Y-%m-%d %H:%M:%S.%f')

        windows = transforms.extract_clean_windows(track, sample_rate, 8, 2)

        self.assertEqual(102, len(windows))