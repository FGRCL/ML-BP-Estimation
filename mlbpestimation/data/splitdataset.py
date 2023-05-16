from pathlib import Path

from tensorflow.python.data import Dataset


class SplitDataset:
    def __init__(self, train: Dataset, validation: Dataset, test: Dataset):
        self.train = train
        self.validation = validation
        self.test = test

    def __iter__(self):
        return iter([self.train, self.validation, self.test])

    def save(self, database_directory: Path):
        self.train.save(str(database_directory / 'train'))
        self.validation.save(str(database_directory / 'validation'))
        self.test.save(str(database_directory / 'test'))
