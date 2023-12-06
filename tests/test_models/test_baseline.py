from os import unlink
from unittest import TestCase

from numpy.testing import assert_allclose

from mlbpestimation.models.tazarv import Tazarv
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestTazarv(TestCase):
    weights_file = "tazarv.keras"

    @classmethod
    def tearDownClass(cls) -> None:
        unlink(cls.weights_file)

    def test_save_model(self):
        model = Tazarv()
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        inputs = next(iter(train.batch(5).take(1)))[0]
        outputs = model(inputs)

        model.save_weights(self.weights_file)
        model.load_weights(self.weights_file)
        result = model(inputs)

        assert_allclose(outputs, result)
