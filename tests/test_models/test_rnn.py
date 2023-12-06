from os import unlink
from unittest import TestCase

from numpy.testing import assert_allclose

from mlbpestimation.models.rnn import Rnn
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestRnn(TestCase):
    weights_file = 'rnn.keras'

    @classmethod
    def tearDownClass(cls) -> None:
        unlink(cls.weights_file)

    def test_create_model(self):
        model = Rnn(1000, 3, .0, .0, 1)

        self.assertIsNotNone(model)

    def test_save_model(self):
        model = Rnn(1000, 3, .0, .0, 1)
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        train = train.batch(5)
        inputs = next(iter(train))[0]
        model.set_input(train.element_spec[:-1])
        model.set_output(train.element_spec[-1])
        outputs = model(inputs)

        model.save_weights(self.weights_file)
        model.load_weights(self.weights_file)
        result = model(inputs)

        assert_allclose(outputs, result, atol=1e-5)
