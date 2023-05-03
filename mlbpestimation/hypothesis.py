from pathlib import Path

import wandb
from keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.models import Model
from wandb import Settings, init
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from mlbpestimation.configuration import configuration
from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.mimic4.mimicdatasetloader import MimicDatasetLoader
from mlbpestimation.data.preprocessed.saveddatasetloader import SavedDatasetLoader
from mlbpestimation.data.preprocessedloader import PreprocessedLoader
from mlbpestimation.data.uci.ucidatasetloader import UciDatasetLoader
from mlbpestimation.data.vitaldb.vitaldatasetloader import VitalDatasetLoader
from mlbpestimation.metrics.standardeviation import AbsoluteError, StandardDeviation
from mlbpestimation.models.baseline import Baseline
from mlbpestimation.models.resnet import ResNet
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


class Hypothesis:
    def __init__(self, dataset_loader: DatasetLoader, model: Model):
        self.dataset_loader = dataset_loader
        self.model = model

    def train(self):
        init(project=configuration['wandb.project_name'],
             entity=configuration['wandb.entity'],
             config=configuration['wandb.config'],
             mode=configuration['wandb.mode'],
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
        self.model.compile(optimizer='Adam', loss=loss, metrics=metrics)

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
                filepath=Path(configuration['output.models']) / wandb.run.name
            )
        ]


hypotheses_repository = {
    'baseline_window_mimic': Hypothesis(PreprocessedLoader(MimicDatasetLoader(), WindowPreprocessing(63)),
                                        Baseline(63)),
    'baseline_window_vitaldb': Hypothesis(PreprocessedLoader(VitalDatasetLoader(), WindowPreprocessing(500)),
                                          Baseline(500)),
    'baseline_heartbeat_mimic': Hypothesis(PreprocessedLoader(MimicDatasetLoader(), HeartbeatPreprocessing(63)),
                                           Baseline(63)),
    'baseline_heartbeat_vitaldb': Hypothesis(PreprocessedLoader(VitalDatasetLoader(), HeartbeatPreprocessing(500)),
                                             Baseline(500)),
    'baseline_window_mimic_preprocessed': Hypothesis(SavedDatasetLoader('mimic-window'), Baseline(63)),
    'resnet_window_mimic_preprocessed': Hypothesis(SavedDatasetLoader('mimic-window'), ResNet()),
    'baseline_window_uci': Hypothesis(PreprocessedLoader(UciDatasetLoader(), HeartbeatPreprocessing(125)),
                                      Baseline(125))
}
