from tensorflow import Tensor

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class RegressionUndersampling(DatasetLoader):
    def __init__(self, dataset_loader: DatasetLoader, dbp_threshold: int, sbp_threshold: int, sampling_ratio: float):
        self.dataset_loader = dataset_loader
        self.dbp_threshold = dbp_threshold
        self.sbp_threshold = sbp_threshold
        self.sampling_ratio = sampling_ratio

    def load_datasets(self) -> SplitDataset:
        train, validation, test = self.dataset_loader.load_datasets()
        majority = train.filter(self._pressures_within_bounds)
        minority = train.filter(self._pressure_out_of_bounds)

        n_resampled = self._compute_n_resampled(minority)
        majority_resampled = self._resample_majority(majority, n_resampled)
        train_resampled = self._merge(majority_resampled, minority)

        return SplitDataset(train_resampled, validation, test)

    def _pressures_within_bounds(self, input_signal: Tensor, pressures: Tensor) -> bool:
        sbp = pressures[0]
        dbp = pressures[1]
        return (self.dbp_threshold < dbp) & (sbp < self.sbp_threshold)

    def _pressure_out_of_bounds(self, input_signal: Tensor, pressures: Tensor) -> bool:
        return not self._pressures_within_bounds(input_signal, pressures)

    def _compute_n_resampled(self, minority):
        n_minority = self._count_dataset(minority)
        n_resampled = int(float(n_minority) / self.sampling_ratio)
        return n_resampled

    def _resample_majority(self, majority, n_resampled):
        n_majority = self._count_dataset(majority)
        majority = majority.shuffle(n_majority)
        majority_resampled = majority.take(n_resampled)
        return majority_resampled

    def _merge(self, majority_resampled, minority):
        train_resampled = majority_resampled.concatenate(minority)
        train_n_resampled = self._count_dataset(train_resampled)
        train_resampled = train_resampled.shuffle(train_n_resampled)
        return train_resampled

    @staticmethod
    def _count_dataset(dataset):
        return int(dataset.reduce(0, lambda x, _: x + 1))
