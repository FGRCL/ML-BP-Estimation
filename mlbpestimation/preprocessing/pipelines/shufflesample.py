from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, Shuffle
from mlbpestimation.preprocessing.shared.transforms import Sample


class ShuffleSample(DatasetPreprocessingPipeline):
    def __init__(self, sample_rate: float):
        super().__init__([
            Shuffle(),
            Sample(sample_rate),
        ])
