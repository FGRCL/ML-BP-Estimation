from abc import ABC, abstractmethod

from mlbpestimation.data.splitdataset import SplitDataset


class DatasetLoader(ABC):

    @abstractmethod
    def load_datasets(self) -> SplitDataset:
        ...
