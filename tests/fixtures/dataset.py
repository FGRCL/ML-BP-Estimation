from neurokit2 import ppg_simulate
from numpy import empty, ndarray
from tensorflow.python.data import Dataset

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class DatasetLoaderFixture(DatasetLoader):
    def __init__(self):
        self.frequency = 125
        self.sample_length = 120

    def load_datasets(self) -> SplitDataset:
        train_samples = self._generate_samples(100)
        validation_samples = self._generate_samples(20)
        test_samples = self._generate_samples(20)

        train_dataset = self._samples_to_dataset(train_samples)
        validation_dataset = self._samples_to_dataset(validation_samples)
        test_dataset = self._samples_to_dataset(test_samples)

        return SplitDataset(train_dataset, validation_dataset, test_dataset)

    def _generate_samples(self, n_samples: int):
        sample_length_frequency = self.frequency * self.sample_length
        samples = empty((n_samples, 2, sample_length_frequency))
        for i in range(n_samples):
            samples[i, 0] = ppg_simulate(sampling_rate=125, frequency_modulation=0.3, ibi_randomness=0.2, drift=1, random_state=1337)
            samples[i, 1] = (ppg_simulate(sampling_rate=125, frequency_modulation=0.1, ibi_randomness=0, drift=0, random_state=1337) + 0.5) * 70 + 35
        return samples

    @staticmethod
    def _samples_to_dataset(samples: ndarray):
        return Dataset.from_tensor_slices((samples[:, 0], samples[:, 1]))
