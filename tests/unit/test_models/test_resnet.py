from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.models.resnet import ResNet
from tests.unit.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestResNet(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('resnet', ignore_errors=True)

    def test_save_model(self):
        model = ResNet(64, 256, 1, 1, 1, 1, 1, 100, 0, 'relu', 0.01, 2, False)
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        inputs = next(iter(train.batch(5).take(1)))[0]
        outputs = model(inputs)

        model.save('resnet', overwrite=True)
        loaded_model: ResNet = load_model('resnet', custom_objects={'ResNet': ResNet})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
