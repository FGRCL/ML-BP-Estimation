from pathlib import Path

import wandb
from omegaconf import DictConfig
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.python.keras.models import Model
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.metrics.maskedmetric import MaskedMetric


class Hypothesis:
    def __init__(self, dataset_loader: DatasetLoader, model: Model, output_directory: str, optimization: DictConfig):
        self.dataset_loader = dataset_loader
        self.model = model
        self.optimization = optimization
        self.output_directory = str(output_directory)
        self.metrics = [
            [
                MaskedMetric(MeanAbsoluteError(), [True, False], name='SBP Mean Absolute Error'),
                # StandardDeviation(AbsoluteError(), name='SBP Absolute Error standard Deviation'),
                MaskedMetric(MeanSquaredError(), [True, False], name='SBP Mean Squared Error'),
                # MeanPrediction(name='SBP Prediction Mean'),
                # StandardDeviation(Prediction(), name='SBP Prediction Standard Deviation')
            ],
            [
                MaskedMetric(MeanAbsoluteError(), [False, True], name='DBP Mean Absolute Error'),
                # StandardDeviation(AbsoluteError(), name='DBP Absolute Error standard Deviation'),
                MaskedMetric(MeanSquaredError(), [False, True], name='DBP Mean Squared Error'),
                # MeanPrediction(name='DBP Prediction Mean'),
                # StandardDeviation(Prediction(), name='DBP Prediction Standard Deviation')
            ]
        ]

    def train(self):
        train, validation = self.setup_train_val()
        self.model.compile(self.optimization.optimizer, loss=self.optimization.loss, metrics=self.metrics)
        self.model.fit(train, epochs=self.optimization.epoch, callbacks=[*self._get_wandb_callbacks()], validation_data=validation)

    def setup_train_val(self):
        datasets = self.dataset_loader.load_datasets()
        train = datasets.train \
            .batch(self.optimization.batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE) \
            .prefetch(AUTOTUNE)
        validation = datasets.validation \
            .batch(self.optimization.batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE)
        return train, validation

    def _get_wandb_callbacks(self):
        return [
            WandbMetricsLogger(
                log_freq="batch"
            ),
            WandbModelCheckpoint(
                filepath=Path(self.output_directory) / wandb.run.name
            )
        ]
