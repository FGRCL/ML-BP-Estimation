from pathlib import Path

from tensorflow.python.data.experimental import load

from mlbpestimation.data.datasource.database import Database
from mlbpestimation.data.multipartdataset import MultipartDataset


class PreprocessedDataSource(Database):
    def get_datasets(self, path: Path):
        train = load(path / 'train')
        validation = load(path / 'validation')
        test = load(path / 'test')

        return MultipartDataset(train, validation, test)
