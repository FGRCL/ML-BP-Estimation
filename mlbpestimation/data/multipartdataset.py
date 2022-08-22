from dataclasses import dataclass

from tensorflow.python.data import Dataset


@dataclass
class MultipartDataset:
    train: Dataset
    validation: Dataset
    test: Dataset

    def __iter__(self):
        return iter([self.train, self.validation, self.test])
