from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset
from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline


class TrainingPipelinePreprocess(DatasetLoader):
    def __init__(self, dataset_loader: DatasetLoader, pipeline: DatasetPreprocessingPipeline):
        self.dataset_loader = dataset_loader
        self.pipeline = pipeline

    def load_datasets(self) -> SplitDataset:
        train, val, test = self.dataset_loader.load_datasets()
        train = self.pipeline.apply(train)
        return SplitDataset(train, val, test)
