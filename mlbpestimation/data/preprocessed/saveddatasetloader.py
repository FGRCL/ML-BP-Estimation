from pathlib import Path

from tensorflow.python.data import Dataset

from mlbpestimation.configuration import configuration
from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class SavedDatasetLoader(DatasetLoader):
    def __init__(self, database_name: str):
        self.directory_path = Path(configuration.directories.data) / database_name

    def load_datasets(self):
        train = Dataset.load(str(self.directory_path / 'train'))
        validation = Dataset.load(str(self.directory_path / 'validation'))
        test = Dataset.load(str(self.directory_path / 'test'))

        return SplitDataset(train, validation, test)
