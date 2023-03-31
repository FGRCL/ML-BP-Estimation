import unittest

from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.optimizer_v2.adam import Adam

from mlbpestimation.data.mimic4 import MimicDatabase
from mlbpestimation.models.baseline import build_baseline_model


class IntegrationTests(unittest.TestCase):

    def test_train_vitaldb_dataset_baseline_model(self):
        train, validate, test = MimicDatabase().load_datasets()
        train = train.take(1)

        (train, validate, test), model = build_baseline_model([train, validate, test], 500)
        train = train.take(100)

        model.summary()
        model.compile(optimizer='Adam', loss=MeanSquaredError(), run_eagerly=True)
        model.fit(train, epochs=1, validation_data=validate)

        predictions = model(test)
        self.assertIsNotNone(predictions)

    def test_train_mimic_dataset_baseline_model(self):
        train, validate, test = MimicDatabase().load_datasets()
        train = train.take(1)

        (train, validate, test), model = build_baseline_model([train, validate, test], 64, 8)
        train = train.take(100)

        model.summary()
        model.compile(optimizer=Adam(), loss=MeanSquaredError(), run_eagerly=True)
        model.fit(train, epochs=1, validation_data=validate)

        predictions = model(test)
        self.assertIsNotNone(predictions)
