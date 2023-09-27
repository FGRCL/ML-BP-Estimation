from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset
from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline


class PreprocessedLoader(DatasetLoader):
    def __init__(self,
                 dataset_loader: DatasetLoader,
                 training_preprocessing: DatasetPreprocessingPipeline,
                 test_preprocessing: DatasetPreprocessingPipeline = None):
        self.dataset_loader = dataset_loader
        self.training_preprocessing = training_preprocessing
        self.test_preprocessing = test_preprocessing if test_preprocessing else training_preprocessing

    def load_datasets(self) -> SplitDataset:
        datasets = self.dataset_loader.load_datasets()
        pipelines = [self.training_preprocessing, self.test_preprocessing, self.test_preprocessing]

        processed_datasets = (pipeline.apply(dataset) for pipeline, dataset in zip(pipelines, datasets))
        return SplitDataset(*processed_datasets)
