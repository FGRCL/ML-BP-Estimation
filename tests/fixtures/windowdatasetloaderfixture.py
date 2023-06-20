from numpy import ones
from tensorflow.python.data import Dataset

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class WindowDatasetLoaderFixture(DatasetLoader):
    def __init__(self):
        self.frequency = 125
        self.sample_length = 8

    def load_datasets(self) -> SplitDataset:
        train_dataset = self._generate_samples(100)
        validation_dataset = self._generate_samples(20)
        test_dataset = self._generate_samples(20)

        return SplitDataset(train_dataset, validation_dataset, test_dataset)

    def _generate_samples(self, n_samples: int):
        frequency_sample_length = self.frequency * self.sample_length
        signal = ones((n_samples, frequency_sample_length, 1))
        pressures = ones((n_samples, 2))
        return Dataset.from_tensor_slices((signal, pressures))
