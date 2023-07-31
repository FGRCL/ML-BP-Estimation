from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline
from mlbpestimation.preprocessing.shared.filters import HasData
from mlbpestimation.preprocessing.shared.transforms import RemoveNan


class FilterHasSignal(DatasetPreprocessingPipeline):
    def __init__(self):
        super().__init__([
            HasData(),
            RemoveNan(),
            HasData(),
        ])
