from tensorflow import Tensor, float32, reshape

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, TransformOperation
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import SignalFilter, StandardizeInput

SECONDS_IN_HOURS = 3600


class SeriesPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int, lowpass_cutoff: int, bandpass_cutoff):
        super().__init__([
            FilterHasSignal(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            StandardizeInput(),
            ExpandFeatureDimension(),
        ])


class ExpandFeatureDimension(TransformOperation):
    def transform(self, input_signal: Tensor, output_signal: Tensor) -> (Tensor, Tensor):
        return reshape(input_signal, (-1, 1)), reshape(output_signal, (-1, 1))
