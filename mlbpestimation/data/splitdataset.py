from pathlib import Path

from tensorflow.python.data import Dataset

from mlbpestimation.configuration import configuration


class SplitDataset:
    def __init__(self, train: Dataset, validation: Dataset, test: Dataset):
        self.train = train
        self.validation = validation
        self.test = test

    def __iter__(self):
        return iter([self.train, self.validation, self.test])

    def save(self, database_name: Path):
        database_path = Path(configuration.directories.data) / database_name
        self.train.save(str(database_path / 'train'))
        self.validation.save(str(database_path / 'validation'))
        self.test.save(str(database_path / 'test'))
