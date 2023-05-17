from pathlib import Path

import wandb
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.metrics import MeanAbsoluteError
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.metrics.standardeviation import AbsoluteError, StandardDeviation


class Hypothesis:
    def __init__(self, dataset_loader: DatasetLoader, model: Model, output_directory: str):
        self.dataset_loader = dataset_loader
        self.model = model
        self.output_directory = str(output_directory)

    def train(self):
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

    def _get_wandb_callbacks(self):
        return [
            WandbMetricsLogger(
                log_freq="batch"
            ),
            WandbModelCheckpoint(
                filepath=Path(self.output_directory) / wandb.run.name
            )
        ]
