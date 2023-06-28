from keras.utils import split_dataset

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class ShuffleSplitDatasetLoader(DatasetLoader):
    def __init__(self, dataset_loader: DatasetLoader, train_size: float, validation_size: float, test_size: float, random_seed: int):
        self.random_seed = random_seed
        self.test_size = test_size
        self.validation_size = validation_size
        self.train_size = train_size
        self.dataset_loader = dataset_loader

    def load_datasets(self) -> SplitDataset:
        train, validation, test = self.dataset_loader.load_datasets()
        merged_dataset = train.concatenate(validation).concatenate(test)
        train_validation_size = 1 - self.test_size
        train_validation, test = split_dataset(merged_dataset, train_validation_size, self.test_size, True, self.random_seed)
        train, validation = split_dataset(train_validation, self.train_size / train_validation_size, self.validation_size / train_validation_size, True,
                                          self.random_seed)
        return SplitDataset(train, validation, test)
