from unittest import TestCase

from mlbpestimation.models.alexnet import AlexNet
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestAlexNet(TestCase):

    def test_can_predict(self):
        model = AlexNet()
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        inputs = next(iter(train.batch(5).take(1)))[0]
        output = model(inputs)

        self.assertEqual(output.shape, [5, 2])
