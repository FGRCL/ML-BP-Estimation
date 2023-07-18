from tensorflow import float32

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline
from mlbpestimation.preprocessing.shared.filters import FilterSqi, HasData
from mlbpestimation.preprocessing.shared.transforms import ComputeSqi, RemoveNan, RemoveSqi


class SqiFiltering(DatasetPreprocessingPipeline):
    def __init__(self, min_sqi, max_sqi, axis=0):
        super().__init__([
            ComputeSqi((float32, float32, float32), axis),
            FilterSqi(min_sqi, max_sqi),
            RemoveSqi(),
        ])


class FilterHasSignal(DatasetPreprocessingPipeline):
    def __init__(self):
        super().__init__([
            HasData(),
            RemoveNan(),
            HasData(),
        ])
