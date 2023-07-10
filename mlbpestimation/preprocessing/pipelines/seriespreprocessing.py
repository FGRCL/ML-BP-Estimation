from tensorflow import Tensor, float32, reshape

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, Print, TransformOperation
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, HasData
from mlbpestimation.preprocessing.shared.pipelines import SqiFiltering
from mlbpestimation.preprocessing.shared.transforms import RemoveNan, SignalFilter, StandardizeArray


class SeriesPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int, lowpass_cutoff: int, bandpass_cutoff, min_pressure, max_pressure):
        super().__init__([
            HasData(),
            RemoveNan(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            SqiFiltering(0.35, 0.8),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            Print("After pressure withing bounds"),
            StandardizeArray(),
            ExpandFeatureDimension(),
        ])


class ExpandFeatureDimension(TransformOperation):
    def transform(self, input_signal: Tensor, output_signal: Tensor) -> (Tensor, Tensor):
        return reshape(input_signal, (-1, 1)), reshape(output_signal, (-1, 1))
