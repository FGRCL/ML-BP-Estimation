from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.data.preprocessed.saveddatasetloader import SavedDatasetLoader
from mlbpestimation.models.mlp import MLP
from tests.constants import data_directory


class TestMLP(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('mlp')

    def test_save_model(self):
        model = MLP(2, 100, 2, 'relu')
        train, _, _ = SavedDatasetLoader(data_directory / 'mimic-window').load_datasets()
        sample = next(iter(train.batch(5).take(1)))
        inputs = sample[0]
        model.set_input_shape(sample)
        outputs = model(inputs)

        model.save('mlp')
        loaded_model: MLP = load_model('mlp', custom_objects={'MLP': MLP})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
