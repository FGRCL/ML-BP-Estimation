from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.models.rnnmlp import RnnMlp
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestRnnMlp(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('rnnmlp', ignore_errors=True)

    def test_create_model(self):
        model = RnnMlp(True, 5, 100, 'gru', 5, 200, 'relu', 2)

        self.assertIsNotNone(model)

    def test_create_model_rnn_after(self):
        model = RnnMlp(False, 5, 100, 'gru', 5, 200, 'relu', 2)

        self.assertIsNotNone(model)

    def test_create_model_lstm(self):
        model = RnnMlp(False, 5, 100, 'lstm', 5, 200, 'relu', 2)

        self.assertIsNotNone(model)

    def test_save_model(self):
        save_directory = 'rnnmlp'
        model = RnnMlp(True, 5, 100, 'gru', 5, 200, 'relu', 2)
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        sample = next(iter(train.batch(5).take(1)))
        inputs = sample[0]
        model.set_input_shape(sample)
        outputs = model(inputs)

        model.save(save_directory, overwrite=True)
        loaded_model: RnnMlp = load_model(save_directory, custom_objects={'RnnMlp': RnnMlp})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
