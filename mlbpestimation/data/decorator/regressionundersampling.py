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
        # TODO needs rewrite
        train, validation, test = self.dataset_loader.load_datasets()
        majority = train.filter(self._pressures_within_bounds)
        minority = train.filter(lambda input_signal, pressures: not self._pressures_within_bounds(input_signal, pressures))

        minority_n_samples = int(minority.reduce(0, lambda x, _: x + 1))
        majority_n_resampled = int(float(minority_n_samples) / self.sampling_ratio)

        majority_n_samples = int(majority.reduce(0, lambda x, _: x + 1))
        majority = majority.shuffle(majority_n_samples)
        majority_resampled = majority.take(majority_n_resampled)

        train_resampled = majority_resampled.concatenate(minority)
        train_n_resampled = int(train_resampled.reduce(0, lambda x, _: x + 1))
        train_resampled = train_resampled.shuffle(train_n_resampled)

        return SplitDataset(train_resampled, validation, test)

    def _pressures_within_bounds(self, input_signal: Tensor, pressures: Tensor) -> bool:
        sbp = pressures[0]
        dbp = pressures[1]
        return (self.dbp_threshold < dbp) & (sbp < self.sbp_threshold)
