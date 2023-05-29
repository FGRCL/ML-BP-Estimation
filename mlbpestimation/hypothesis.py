from pathlib import Path

import wandb
from keras.callbacks import EarlyStopping
from keras.metrics import MeanAbsoluteError, MeanSquaredError
from omegaconf import DictConfig
from tensorflow.python.data import AUTOTUNE
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.metrics.maskedmetric import MaskedMetric
from mlbpestimation.metrics.meanprediction import MeanPrediction
from mlbpestimation.metrics.standardeviation import StandardDeviationAbsoluteError, StandardDeviationPrediction
from mlbpestimation.models.basemodel import BloodPressureModel


class Hypothesis:
    def __init__(self, dataset: DatasetLoader, model: BloodPressureModel, output_directory: str, optimization: DictConfig):
        self.dataset = dataset
        self.model = model
        self.optimization = optimization
        self.output_directory = str(output_directory)

    def train(self):
        train, validation = self.setup_train_val()
        self.model.set_input_shape(train.element_spec)
        self.model.compile(self.optimization.optimizer, loss=self.optimization.loss, metrics=self._build_metrics())
        self.model.fit(train, epochs=self.optimization.epoch, callbacks=self._build_callbacks(), validation_data=validation)

    def setup_train_val(self):
        datasets = self.dataset.load_datasets()
        train = datasets.train \
            .batch(self.optimization.batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE) \
            .prefetch(AUTOTUNE)
        validation = datasets.validation \
            .batch(self.optimization.batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE)
        if self.optimization.n_batches is not None:
            train = train.take(self.optimization.n_batches)
            validation = validation.take(int(self.optimization.n_batches * 0.15))
        return train, validation

    def _build_callbacks(self):
        return [
            *self._get_wandb_callbacks(),
            EarlyStopping(patience=5)
        ]

    def _get_wandb_callbacks(self):
        return [
            WandbMetricsLogger(
                log_freq="batch"
            ),
            WandbModelCheckpoint(
                filepath=Path(self.output_directory) / wandb.run.name,
                save_best_only=True
            )
        ]

    def _build_metrics(self):
        metric_masks = [
            ([True, False], 'SBP'),
            ([False, True], 'DBP')
        ]
        metrics = []
        for mask, name in metric_masks:
            metrics += [
                MaskedMetric(MeanAbsoluteError(), mask, name=f'{name} Mean Absolute Error'),
                MaskedMetric(StandardDeviationAbsoluteError(), mask, name=f'{name} Absolute Error standard Deviation'),
                MaskedMetric(MeanSquaredError(), mask, name=f'{name} Mean Squared Error'),
                MaskedMetric(MeanPrediction(), mask, name=f'{name} Prediction Mean'),
                MaskedMetric(StandardDeviationPrediction(), mask, name=f'{name} Prediction Standard Deviation'),
            ]
        return metrics
