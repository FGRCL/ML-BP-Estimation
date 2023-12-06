from os import unlink
from unittest import TestCase

from numpy.testing import assert_allclose

from mlbpestimation.models.mlp import MLP
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestMLP(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        unlink('mlp.keras')

    def test_save_model(self):
        model = MLP(2, 100, 2, 'relu')
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        train = train.batch(5)
        inputs = next(iter(train))[0]
        model.set_input(train.element_spec[:-1])
        model.set_output(train.element_spec[-1])
        outputs = model(inputs)

        model.save_weights('mlp.keras')
        model.load_weights('mlp.keras')
        result = model(inputs)

        assert_allclose(outputs, result)
