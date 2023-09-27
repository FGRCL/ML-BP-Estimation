from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.models.slapnicar import Slapnicar
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestSlapnicar(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('slapnicar', ignore_errors=True)

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

        model.save(save_directory, overwrite=True)
        loaded_model: Slapnicar = load_model(save_directory, custom_objects={'Slapnicar': Slapnicar})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
