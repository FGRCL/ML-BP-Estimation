from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.data.decorator.saveddatasetloader import SavedDatasetLoader
from mlbpestimation.models.slapnicar import Slapnicar
from tests.constants import data_directory


class TestSlapnicar(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('slapnicar')

    def test_create_model(self):
        model = Slapnicar(16, 128, 5, [8, 5, 5, 3], 2, 1, 65, 32, 2, .001, .25)

        self.assertIsNotNone(model)

    def test_save_model(self):
        save_directory = 'slapnicar'
        model = Slapnicar(16, 32, 5, [8, 5, 5, 3], 2, 1, 65, 32, 2, .001, .25)
        train, _, _ = SavedDatasetLoader(data_directory / 'mimic-window').load_datasets()
        sample = next(iter(train.batch(5).take(1)))
        inputs = sample[0]
        model.set_input_shape(sample)
        outputs = model(inputs)

        model.save(save_directory, overwrite=True)
        loaded_model: Slapnicar = load_model(save_directory, custom_objects={'Slapnicar': Slapnicar})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
