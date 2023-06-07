from tensorflow import float32

from tensorflow import float32

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline
from mlbpestimation.preprocessing.shared.filters import FilterSqi
from mlbpestimation.preprocessing.shared.transforms import ComputeSqi, RemoveSqi


class SqiFiltering(DatasetPreprocessingPipeline):
    def __init__(self, min_sqi, max_sqi, axis=0):
        super().__init__([
            ComputeSqi((float32, float32, float32), axis),
            FilterSqi(min_sqi, max_sqi),
            RemoveSqi(),
        ])
