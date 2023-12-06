from os import unlink
from unittest import TestCase

from numpy.testing import assert_allclose

from mlbpestimation.models.resnet import ResNet
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestResNet(TestCase):
    weights_file = "resnet.keras"

    @classmethod
    def tearDownClass(cls) -> None:
        unlink(cls.weights_file)

    def test_save_model(self):
        model = ResNet(64, 256, 1, 1, 1, 1, 1, 100, 0, 'relu', 0.01, 2, False)
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
