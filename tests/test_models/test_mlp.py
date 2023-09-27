from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.models.mlp import MLP
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestMLP(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('mlp', ignore_errors=True)

    def test_save_model(self):
        model = MLP(2, 100, 2, 'relu')
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        train = train.batch(5)
        inputs = next(iter(train))[0]
        model.set_input(train.element_spec[:-1])
        model.set_output(train.element_spec[-1])
        outputs = model(inputs)

        model.save('mlp')
        loaded_model: MLP = load_model('mlp', custom_objects={'MLP': MLP})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
