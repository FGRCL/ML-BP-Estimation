from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.models.transformerencoder import TransformerEncoder
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestTransformerEncoder(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('transformerencoder', ignore_errors=True)

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

        model.save(save_directory, overwrite=True)
        loaded_model: TransformerEncoder = load_model(save_directory, custom_objects={'TransformerEncoder': TransformerEncoder})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
