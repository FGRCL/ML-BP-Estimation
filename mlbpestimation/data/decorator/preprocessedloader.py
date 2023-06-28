from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset
from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline


class PreprocessedLoader(DatasetLoader):
    def __init__(self, dataset_loader: DatasetLoader, preprocessing: DatasetPreprocessingPipeline):
        self.dataset_loader = dataset_loader
        self.preprocessing = preprocessing

    def load_datasets(self) -> SplitDataset:
        return SplitDataset(*map(self.preprocessing.preprocess, self.dataset_loader.load_datasets()))
