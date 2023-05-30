from unittest import TestCase

from mlbpestimation.models.transformerencoder import TransformerEncoder


class TestTransformerEncoder(TestCase):
    def test_build_model(self):
        model = TransformerEncoder(5, 2, 100, 0.01, 500, 0.1, 2, 2000, 2, 0.1)

        self.assertIsNotNone(model)
