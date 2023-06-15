from neurokit2 import ppg_simulate
from numpy import empty
from tensorflow.python.data import Dataset

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class DatasetLoaderFixture(DatasetLoader):
    def load_datasets(self) -> SplitDataset:
        frequency = 125
        sample_length = 120
        sample_length_frequency = frequency * sample_length
        n_samples = 100

        samples = empty((n_samples, 2, sample_length_frequency))
        for i in range(n_samples):
            samples[i, 0] = ppg_simulate(sampling_rate=125, frequency_modulation=0.5, ibi_randomness=0.3, drift=2)
            samples[i, 1] = (ppg_simulate(sampling_rate=125, drift=0) + 0.5) * 70 + 35

        return SplitDataset(Dataset.from_tensor_slices((samples[:, 0], samples[:, 1])), None, None)
