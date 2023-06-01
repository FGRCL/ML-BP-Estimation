from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.data.saveddatasetloader import SavedDatasetLoader
from mlbpestimation.models.transformerencoder import TransformerEncoder
from tests.constants import data_directory


class TestTransformerEncoder(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('transformerencoder')

    def test_build_model(self):
        model = TransformerEncoder(5, 2, 100, 0.01, 500, 0.1, 2, 2000, 2, 0.1)

        self.assertIsNotNone(model)

    def test_save_model(self):
        save_directory = 'transformerencoder'
        model = TransformerEncoder(5, 2, 100, 0.01, 500, 0.1, 2, 2000, 2, 0.1)
        train, _, _ = SavedDatasetLoader(data_directory / 'mimic-window').load_datasets()
        sample = next(iter(train.batch(5).take(1)))
        inputs = sample[0]
        model.set_input_shape(sample)
        outputs = model(inputs)

        model.save(save_directory, overwrite=True)
        loaded_model: TransformerEncoder = load_model(save_directory, custom_objects={'TransformerEncoder': TransformerEncoder})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
