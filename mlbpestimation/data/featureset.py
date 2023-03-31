from pathlib import Path

from mlbpestimation.configuration import configuration
from mlbpestimation.data.datasource.database import Database
from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline


class FeatureSet:
    def __init__(self, database: Database, preprocessing: DatasetPreprocessingPipeline = None):
        self._database = database
        self._preprocessing = preprocessing
        self.train = None
        self.validation = None
        self.test = None

    def build_featuresets(self):
        featuresets = []
        for dataset in self._database.get_datasets():
            featureset = self._preprocessing.preprocess(dataset)
            featuresets.append(featureset)

        self.train = featuresets[0]
        self.validation = featuresets[1]
        self.test = featuresets[2]
        return self

    def save(self, database_name: Path):
        database_path = Path(configuration['data.directory']) / database_name
        self.train.save(str(database_path / 'train'))
        self.validation.save(str(database_path / 'validation'))
        self.test.save(str(database_path / 'test'))
