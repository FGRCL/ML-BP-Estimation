from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.data.preprocessed.saveddatasetloader import SavedDatasetLoader
from mlbpestimation.models.baseline import Baseline
from tests.constants import data_directory


class TestBaseline(TestCase):
    def test_save_model(self):
        model = Baseline()
        train, _, _ = SavedDatasetLoader(data_directory / 'mimic-window').load_datasets()
        inputs = next(iter(train.batch(5).take(1)))[0]
        outputs = model(inputs)

        model.save('baseline')
        loaded_model: Baseline = load_model('baseline', custom_objects={'Baseline': Baseline})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
