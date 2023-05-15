from pathlib import Path

import wandb
from keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from wandb import Settings, init
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from mlbpestimation.conf import configuration
from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.metrics.standardeviation import AbsoluteError, StandardDeviation


class Hypothesis:
    def __init__(self, dataset_loader: DatasetLoader, model: Model):
        self.dataset_loader = dataset_loader
        self.model = model

    def train(self):
        init(project=configuration.wandb.project_name,
             entity=configuration.wandb.entity,
             mode=configuration.wandb.mode,
             settings=Settings(start_method='fork'))

        datasets = self.dataset_loader.load_datasets()
        train = datasets.train \
            .batch(20, drop_remainder=True, num_parallel_calls=AUTOTUNE) \
            .prefetch(AUTOTUNE)
        validation = datasets.validation \
            .batch(20, drop_remainder=True, num_parallel_calls=AUTOTUNE)

        loss = MeanSquaredError()
        metrics = [
            MeanAbsoluteError(),
            StandardDeviation(AbsoluteError())
        ]
        self.model.compile(Adam(1e-6), loss=loss, metrics=metrics)

        self.model.fit(train,
                       epochs=100,
                       callbacks=[*self._get_wandb_callbacks()],
                       validation_data=validation)

    @staticmethod
    def _get_wandb_callbacks():
        return [
            WandbMetricsLogger(
                log_freq="batch"
            ),
            WandbModelCheckpoint(
                filepath=Path(configuration.directories.output) / wandb.run.name
            )
        ]
