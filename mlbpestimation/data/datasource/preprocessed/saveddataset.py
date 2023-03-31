from pathlib import Path

from tensorflow.python.data import Dataset

from mlbpestimation.configuration import configuration
from mlbpestimation.data.datasource.database import Database
from mlbpestimation.data.multipartdataset import MultipartDataset


class SavedDataset(Database):
    def __init__(self, database_name: str):
        self.directory_path = Path(configuration['data.directory']) / database_name

    def get_datasets(self):
        train = Dataset.load(str(self.directory_path / 'train'))
        validation = Dataset.load(str(self.directory_path / 'validation'))
        test = Dataset.load(str(self.directory_path / 'test'))

        return MultipartDataset(train, validation, test)
