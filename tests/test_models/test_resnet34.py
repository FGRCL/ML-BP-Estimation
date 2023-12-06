from unittest import TestCase

from mlbpestimation.models.resnet34 import ResNet34
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestResNet34(TestCase):

    def test_can_predict(self):
        model = ResNet34(64, 256, 1, 1, 1, 1, 1, 100, 0, 'relu', 0.01, 2)
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        train = train.batch(5).take(1)
        inputs = next(iter(train))[0]
        model.set_input(train.element_spec[:-1])
        model.set_output(train.element_spec[-1])
        output = model(inputs)

        self.assertEqual(output.shape, [5, 2])