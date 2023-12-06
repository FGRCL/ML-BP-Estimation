from os import unlink
from unittest import TestCase

from numpy.testing import assert_allclose

from mlbpestimation.models.transformerencoder import TransformerEncoder
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestTransformerEncoder(TestCase):
    weights_file = "transformerencoder.keras"

    @classmethod
    def tearDownClass(cls) -> None:
        unlink(cls.weights_file)

    def test_build_model(self):
        model = TransformerEncoder(5, 2, 0.01, 500, 0.1, 2, 2000, 2, 0.1)

        self.assertIsNotNone(model)

    def test_save_model(self):
        save_directory = 'transformerencoder'
        model = TransformerEncoder(5, 2, 0.01, 500, 0.1, 2, 2000, 2, 0.1)
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
