from keras.losses import MeanAbsoluteError, MeanSquaredError
from wandb import Settings, init
from wandb.integration.keras import WandbCallback

from mlbpestimation.configuration import configuration
from mlbpestimation.data.datasource.mimic4.dataset import MimicDataSource
from mlbpestimation.data.datasource.vitaldb.casesplit import VitalDBDataSource
from mlbpestimation.data.featureset import FeatureSet
from mlbpestimation.metrics.standardeviation import AbsoluteError, StandardDeviation
from mlbpestimation.models.baseline import Baseline
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
             settings=Settings(start_method='fork'),
             name=configuration['wandb.run_name'])

        self.featureset.build_featuresets(20)
        # self.model.build()
        # self.model.summary()
        self.model.compile(optimizer='Adam',
                           loss=MeanSquaredError(),
                           metrics=[
                               MeanAbsoluteError(),
                               StandardDeviation(AbsoluteError())
                           ])
        self.model.fit(self.featureset.train,
                       epochs=100,
                       callbacks=[WandbCallback()],
                       validation_data=self.featureset.validation)


hypotheses_repository = {
    'baseline': Hypothesis(FeatureSet(MimicDataSource(), WindowPreprocessing()), Baseline()),
    'baseline_vitaldb': Hypothesis(FeatureSet(VitalDBDataSource(), WindowPreprocessing()), Baseline())
}
