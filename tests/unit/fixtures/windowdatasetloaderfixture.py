from random import randint, seed

from neurokit2 import ppg_simulate
from numpy import empty, mean, std
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
        input = empty((n_samples, frequency_sample_length, 1))
        pressures = empty((n_samples, 2))
        seed(133)
        for i in range(n_samples):
            signal = ppg_simulate(duration=self.sample_length, sampling_rate=self.frequency, frequency_modulation=0.3, ibi_randomness=0.2, drift=1,
                                  random_state=randint(0, 10000)) * 30 + 70
            sbp = max(signal)
            dbp = min(signal)
            signal_rescaled = (signal - mean(signal)) / std(signal)

            input[i, :, :] = signal_rescaled.reshape((-1, 1))
            pressures[i, 0] = sbp
            pressures[i, 1] = dbp

        return Dataset.from_tensor_slices((input, pressures))
