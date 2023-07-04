from pathlib import Path

from tensorflow.python.data import Dataset

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class TFRecordDatasetLoader(DatasetLoader):
    def __init__(self, dataset_directory: str):
        self.dataset_directory = Path(dataset_directory)

    def load_datasets(self):
        train = Dataset.load(str(self.dataset_directory / 'train'))
        validation = Dataset.load(str(self.dataset_directory / 'validation'))
        test = Dataset.load(str(self.dataset_directory / 'test'))

        return SplitDataset(train, validation, test)
