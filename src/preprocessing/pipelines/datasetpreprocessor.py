from abc import ABC, abstractmethod

from tensorflow.python.data import Dataset


class DatasetPreprocessor(ABC):

    @abstractmethod
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        pass