from os import unlink
from unittest import TestCase

from numpy.testing import assert_allclose

from mlbpestimation.models.slapnicar import Slapnicar
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestSlapnicar(TestCase):
    weights_file = "slapnicar.keras"

    @classmethod
    def tearDownClass(cls) -> None:
        unlink(cls.weights_file)

    def test_create_model(self):
        model = Slapnicar(16, 128, 5, [8, 5, 5, 3], 2, 1, 65, 32, 2, .001, .25)

        self.assertIsNotNone(model)

    def test_save_model(self):
        save_directory = 'slapnicar'
        model = Slapnicar(16, 32, 5, [8, 5, 5, 3], 2, 1, 65, 32, 2, .001, .25)
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        train = train.batch(5)
        inputs = next(iter(train))[0]
        model.set_input(train.element_spec[:-1])
        model.set_output(train.element_spec[-1])
        outputs = model(inputs)

        model.save_weights(self.weights_file)
        model.load_weights(self.weights_file)
        result = model(inputs)

        assert_allclose(outputs, result)
