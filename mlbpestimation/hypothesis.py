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
from mlbpestimation.metrics.meanprediction import MeanPrediction
from mlbpestimation.metrics.standardeviation import StandardDeviationAbsoluteError, StandardDeviationPrediction
from mlbpestimation.metrics.thresholdmetric import ThresholdMetric
from mlbpestimation.metrics.totalmeanabsoluteerror import TotalMeanAbsoluteErrorMetric
from mlbpestimation.metrics.transformmetric import TransformMetric
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
        log.info(train)
        self.model.set_input(train.element_spec[:-1])
        self.model.set_output(train.element_spec[-1])
        self.model.compile(self.optimization.optimizer, loss=self.optimization.loss, metrics=self._build_metrics())
        self.model.build([spec.shape for spec in train.element_spec[0]])  # TODO keep this?
        self.model.summary()
        self.model.fit(train, epochs=self.optimization.epoch, callbacks=self._build_training_callbacks(), validation_data=validation)
        log.info('Finished training')

    def evaluate(self):
        log.info('Start evaluation')

        test = self.dataset.load_datasets().test \
            .batch(self.optimization.batch_size) \
            .prefetch(AUTOTUNE)

        if self.optimization.n_batches is not None:
            test = test.take(int(self.optimization.n_batches * 0.15))

        self.model.evaluate(test, callbacks=self._build_evaluation_callbacks())
        log.info('Finished evaluation')

    def setup_train_val(self):
        train, validation, _ = self.dataset.load_datasets()

        train = train \
            .batch(self.optimization.batch_size) \
            .prefetch(AUTOTUNE)

        validation = validation \
            .batch(self.optimization.batch_size) \
            .prefetch(AUTOTUNE)

        if self.optimization.n_batches is not None:
            train = train.take(self.optimization.n_batches)
            validation = validation.take(int(self.optimization.n_batches * 0.15))
        return train, validation

    def _build_training_callbacks(self):
        return [
            WandbMetricsLogger(
                log_freq="batch"
            ),
            WandbModelCheckpoint(
                filepath=Path(self.output_directory) / wandb.run.name / '{epoch:02d}',
                save_best_only=True
            ),
            EarlyStopping()
        ]

    @staticmethod
    def _build_evaluation_callbacks():
        return [
            EvaluateCallback()
        ]

    def _build_metrics(self):
        reducer = self.model.get_metric_reducer_strategy()
        metrics = [
            TransformMetric(MeanAbsoluteError(), reducer.pressure_reduce, name='Mean Absolute Error'),
            TransformMetric(TotalMeanAbsoluteErrorMetric(), reducer.pressure_reduce, name='Total Mean Absolute Error'),
        ]

        for reduce, name in [(reducer.sbp_reduce, 'SBP'), (reducer.dbp_reduce, 'DBP')]:
            metrics += [
                TransformMetric(MeanAbsoluteError(), reduce, name=f'{name} Mean Absolute Error'),
                TransformMetric(StandardDeviationAbsoluteError(), reduce, name=f'{name} Absolute Error standard Deviation'),
                TransformMetric(ThresholdMetric(MeanAbsoluteError(), upper=50), reduce, name=f'{name} Mean Absolute Error under 50mmHg'),
                TransformMetric(ThresholdMetric(MeanAbsoluteError(), lower=150), reduce, name=f'{name} Mean Absolute Error above 150mmHg'),
                TransformMetric(ThresholdMetric(MeanAbsoluteError(), lower=50, upper=150), reduce,
                                name=f'{name} Mean Absolute Error between 50mmHg and 150mmHg'),
                TransformMetric(MeanSquaredError(), reduce, name=f'{name} Mean Squared Error'),
                TransformMetric(MeanPrediction(), reduce, name=f'{name} Prediction Mean'),
                TransformMetric(StandardDeviationPrediction(), reduce, name=f'{name} Prediction Standard Deviation'),

            ]
        return metrics
