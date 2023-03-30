from keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.python.data import AUTOTUNE
from wandb import Settings, init
from wandb.integration.keras import WandbCallback

from mlbpestimation.configuration import configuration
from mlbpestimation.data.datasource.mimic4.mimicdatabase import MimicDatabase
from mlbpestimation.data.datasource.vitaldb.vitaldatabase import VitalDatabase
from mlbpestimation.data.featureset import FeatureSet
from mlbpestimation.metrics.standardeviation import AbsoluteError, StandardDeviation
from mlbpestimation.models.baseline import Baseline
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


class Hypothesis:
    def __init__(self, featureset, model):
        self.featureset = featureset
        self.model = model

    def train(self):
        init(project=configuration['wandb.project_name'],
             entity=configuration['wandb.entity'],
             config=configuration['wandb.config'],
             mode=configuration['wandb.mode'],
             settings=Settings(start_method='fork'))

        self.featureset.build_featuresets()
        train_set = self.featureset.train \
            .batch(20, drop_remainder=True, num_parallel_calls=AUTOTUNE) \
            .prefetch(AUTOTUNE)
        self.model.compile(optimizer='Adam',
                           loss=MeanSquaredError(),
                           metrics=[
                               MeanAbsoluteError(),
                               StandardDeviation(AbsoluteError())
                           ])
        self.model.fit(train_set,
                       epochs=100,
                       callbacks=[WandbCallback()],
                       validation_data=self.featureset.validation)


hypotheses_repository = {
    'baseline_window_mimic': Hypothesis(FeatureSet(MimicDatabase(), WindowPreprocessing(63)), Baseline(63)),
    'baseline_window_vitaldb': Hypothesis(FeatureSet(VitalDatabase(), WindowPreprocessing(500)), Baseline(500)),
    'baseline_heartbeat_mimic': Hypothesis(FeatureSet(MimicDatabase(), HeartbeatPreprocessing(63)), Baseline(63)),
    'baseline_heartbeat_vitaldb': Hypothesis(FeatureSet(VitalDatabase(), HeartbeatPreprocessing(500)), Baseline(500)),
}
