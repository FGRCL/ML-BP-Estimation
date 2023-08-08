from typing import Optional

from tensorflow.python.data import Dataset

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class DatasetLoaderFixture(DatasetLoader):
    def __init__(self, train: Dataset, validation: Optional[Dataset], test: Optional[Dataset]):
        self.train = train
        self.validation = validation
        self.test = test

    def load_datasets(self) -> SplitDataset:
        return SplitDataset(self.train, self.validation, self.test)
