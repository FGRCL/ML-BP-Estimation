from pathlib import Path

from tensorflow.python.data import AUTOTUNE

from mlbpestimation.data.datasource.database import Database
from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline


class FeatureSet:
    def __init__(self, database: Database, preprocessing: DatasetPreprocessingPipeline = None):
        self._database = database
        self._preprocessing = preprocessing
        self.train = None
        self.validation = None
        self.test = None

    def build_featuresets(self, batch_size):
        featuresets = []
        for dataset in self._database.get_datasets():
            featureset = self._preprocessing.preprocess(dataset) \
                .batch(batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE) \
                .prefetch(AUTOTUNE)
            featuresets.append(featureset)

        self.train = featuresets[0]
        self.validation = featuresets[1]
        self.test = featuresets[2]
        return self

    def save(self, path: Path):
        self.train.save(str(path / 'train'))
        self.validation.save(str(path / 'validation'))
        self.test.save(str(path / 'test'))
