import unittest

from mlbpestimation.data.vitaldb.casesplit import get_splits


class TestCaseSplit(unittest.TestCase):

    def test_splitgenerator(self):
        result = get_splits(range(0, 100), [0.5, 0.5])

        print(len(result[0]))

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 50)
        self.assertEqual(len(result[1]), 50)

        for i in range(0, len(result)):
            for j in range(i + 1, len(result)):
                self.assertTrue(set(result[i]).isdisjoint(set(result[j])))

    def test_splitgenerator(self):
        result = get_splits(range(0, 1000), [0.7, 0.15, 0.15])

        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 700)
        self.assertEqual(len(result[1]), 150)
        self.assertEqual(len(result[2]), 150)

        for i in range(0, len(result)):
            for j in range(i + 1, len(result)):
                self.assertTrue(set(result[i]).isdisjoint(set(result[j])))

    def test_splitgenerator(self):
        result = get_splits(range(0, 1000), [0.1] * 10)

        self.assertEqual(len(result), 10)
        for l in result:
            self.assertTrue(98 <= len(l) <= 102)

        for i in range(0, len(result)):
            for j in range(i + 1, len(result)):
                self.assertTrue(set(result[i]).isdisjoint(set(result[j])))
