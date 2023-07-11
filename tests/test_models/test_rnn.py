from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.models.rnn import Rnn
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestRnn(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('rnn', ignore_errors=True)

    def test_create_model(self):
        model = Rnn(1000, 3, 'rnn', 1)

        self.assertIsNotNone(model)

    def test_save_model(self):
        model = Rnn(1000, 3, 'rnn', 1)
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        sample = next(iter(train.batch(5).take(1)))
        inputs = sample[0]
        model.set_input_shape(sample)
        outputs = model(inputs)

        model.save('rnn', overwrite=True)
        loaded_model: Rnn = load_model('rnn', custom_objects={'Rnn': Rnn})
        result = loaded_model(inputs)

        assert_allclose(outputs, result, atol=1e-5)
