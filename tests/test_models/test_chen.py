from unittest import TestCase

from mlbpestimation.models.chen import Chen
from tests.fixtures.windowdatasetloaderfixture import WindowDatasetLoaderFixture


class TestChen(TestCase):

    def test_can_predict(self):
        model = Chen()
        train, _, _ = WindowDatasetLoaderFixture().load_datasets()
        train = train.batch(5).take(1)
        inputs = next(iter(train))[0]
        model.set_input(train.element_spec[:-1])
        model.set_output(train.element_spec[-1])
        output = model(inputs)

        self.assertEqual(output.shape, [5, 2])
