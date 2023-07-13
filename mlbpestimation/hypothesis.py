import logging
from pathlib import Path

import wandb
from keras.callbacks import EarlyStopping
from keras.metrics import MeanAbsoluteError, MeanSquaredError
from omegaconf import DictConfig
from tensorflow.python.data import AUTOTUNE
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from mlbpestimation.callbacks.evaluatecallback import EvaluateCallback
from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.metrics.maskedmetric import MaskedMetric
from mlbpestimation.metrics.meanprediction import MeanPrediction
from mlbpestimation.metrics.standardeviation import StandardDeviationAbsoluteError, StandardDeviationPrediction
from mlbpestimation.metrics.thresholdmetric import ThresholdMetric
from mlbpestimation.metrics.totalmeanabsoluteerror import TotalMeanAbsoluteErrorMetric
from mlbpestimation.models.basemodel import BloodPressureModel

log = logging.getLogger(__name__)


class Hypothesis:
    def __init__(self, dataset: DatasetLoader, model: BloodPressureModel, output_directory: str, optimization: DictConfig):
        self.dataset = dataset
        self.model = model
        self.optimization = optimization
        self.output_directory = str(output_directory)

    def train(self):
        log.info('Start training')
        train, validation = self.setup_train_val()
        pressure_output = train.element_spec[1].shape[0] == 2
        self.model.set_input_shape(train.element_spec)
        self.model.compile(self.optimization.optimizer, loss=self.optimization.loss, metrics=self._build_metrics(pressure_output))
        self.model.fit(train, epochs=self.optimization.epoch, callbacks=self._build_callbacks(), validation_data=validation)
        log.info('Finished training')

    def evaluate(self):
        log.info('Start evaluation')

        test = self.dataset.load_datasets().test \
            .cache() \
            .padded_batch(self.optimization.batch_size) \
            .prefetch(AUTOTUNE)

        if self.optimization.n_batches is not None:
            test = test.take(int(self.optimization.n_batches * 0.15))

        self.model.evaluate(test, callbacks=self._get_eval_callbacks())
        log.info('Finished evaluation')

    def setup_train_val(self):
        datasets = self.dataset.load_datasets()
        train = datasets.train \
            .cache() \
            .padded_batch(self.optimization.batch_size) \
            .prefetch(AUTOTUNE)

        validation = datasets.validation \
            .cache() \
            .padded_batch(self.optimization.batch_size) \
            .prefetch(AUTOTUNE)

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
                filepath=Path(self.output_directory) / wandb.run.name / '{epoch:02d}',
                save_best_only=True
            )
        ]

    def _get_eval_callbacks(self):
        return [
            EvaluateCallback()
        ]

    def _build_metrics(self, include_pressure_metrics: bool):
        metric_masks = [
            ([True, False], 'SBP'),
            ([False, True], 'DBP')
        ]
        metrics = [
            MeanAbsoluteError(name='Mean Absolute Error'),
            TotalMeanAbsoluteErrorMetric(name='Total Mean Absolute Error')
        ]
        if include_pressure_metrics:
            for mask, name in metric_masks:
                metrics += [
                    MaskedMetric(MeanAbsoluteError(), mask, name=f'{name} Mean Absolute Error'),
                    MaskedMetric(StandardDeviationAbsoluteError(), mask, name=f'{name} Absolute Error standard Deviation'),
                    MaskedMetric(ThresholdMetric(MeanAbsoluteError(), upper=50), mask, name=f'{name} Mean Absolute Error under 50mmHg'),
                    MaskedMetric(ThresholdMetric(MeanAbsoluteError(), lower=150), mask, name=f'{name} Mean Absolute Error above 150mmHg'),
                    MaskedMetric(ThresholdMetric(MeanAbsoluteError(), lower=50, upper=150), mask,
                                 name=f'{name} Mean Absolute Error between 50mmHg and 150mmHg'),
                    MaskedMetric(MeanSquaredError(), mask, name=f'{name} Mean Squared Error'),
                    MaskedMetric(MeanPrediction(), mask, name=f'{name} Prediction Mean'),
                    MaskedMetric(StandardDeviationPrediction(), mask, name=f'{name} Prediction Standard Deviation'),
                ]
        return metrics
