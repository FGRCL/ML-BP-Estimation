from unittest import TestCase

from mlbpestimation.models.slapnicar import Slapnicar


class TestSlapnicar(TestCase):
    def test_create_model(self):
        model = Slapnicar(16, 128, 5, [8, 5, 5, 3], 2, 1, 65, 32, 2, .001, .25)

        self.assertIsNotNone(model)
