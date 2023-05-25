from shutil import rmtree
from unittest import TestCase

from keras.saving.saving_api import load_model
from numpy.testing import assert_allclose

from mlbpestimation.data.preprocessed.saveddatasetloader import SavedDatasetLoader
from mlbpestimation.models.resnet import ResNet
from tests.constants import data_directory


class TestResNet(TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        rmtree('resnet')

    def test_save_model(self):
        model = ResNet([1, 1, 1], [5, 5, 5])
        train, _, _ = SavedDatasetLoader(data_directory / 'mimic-window').load_datasets()
        inputs = next(iter(train.batch(5).take(1)))[0]
        outputs = model(inputs)

        model.save('resnet', overwrite=True)
        loaded_model: ResNet = load_model('resnet', custom_objects={'ResNet': ResNet})
        result = loaded_model(inputs)

        assert_allclose(outputs, result)
