from pathlib import Path

from tensorflow.python.data import Dataset

from mlbpestimation.data.datasource.database import Database
from mlbpestimation.data.multipartdataset import MultipartDataset


class SavedDataset(Database):
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def get_datasets(self):
        train = Dataset.load(self.base_path / 'train')
        validation = Dataset.load(self.base_path / 'validation')
        test = Dataset.load(self.base_path / 'test')

        return MultipartDataset(train, validation, test)
