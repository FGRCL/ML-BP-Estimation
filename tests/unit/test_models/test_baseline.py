from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.models.tazarv import Tazarv
from tests.unit.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestTazarv(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('tazarv', ignore_errors=True)

    def test_save_model(self):
        model = Tazarv()
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        inputs = next(iter(train.batch(5).take(1)))[0]
        outputs = model(inputs)

        model.save('tazarv')
        loaded_model: Tazarv = load_model('tazarv', custom_objects={'Tazarv': Tazarv})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
