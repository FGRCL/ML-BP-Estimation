from os import unlink
from unittest import TestCase

from numpy.testing import assert_allclose

from mlbpestimation.models.rnnmlp import RnnMlp
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestRnnMlp(TestCase):
    weights_file = 'rnnmlp.keras'

    @classmethod
    def tearDownClass(cls) -> None:
        unlink(cls.weights_file)

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
        train = train.batch(5)
        inputs = next(iter(train))[0]
        model.set_input(train.element_spec[:-1])
        model.set_output(train.element_spec[-1])
        outputs = model(inputs)

        model.save_weights(self.weights_file)
        model.load_weights(self.weights_file)
        result = model(inputs)

        assert_allclose(outputs, result)
