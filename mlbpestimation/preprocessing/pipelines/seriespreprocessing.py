from typing import Any, Tuple, Union

from numpy import ndarray
from scipy.signal import resample
from tensorflow import DType, Tensor, float32, reshape

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, NumpyTransformOperation, TransformOperation
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import SignalFilter, SlidingWindow, StandardScaling

SECONDS_IN_MINUTES = 60


class SeriesPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int, resample_frequency: int, window_size: float, lowpass_cutoff: int, bandpass_cutoff):
        window_samples = int(window_size * resample_frequency * SECONDS_IN_MINUTES)
        super().__init__([
            FilterHasSignal(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            ResampleSignals((float32, float32), frequency, resample_frequency),
            SlidingWindow(window_samples, window_samples),
            StandardScaling(axis=1),
            ExpandFeatureDimension(),
        ])


class ExpandFeatureDimension(TransformOperation):
    def transform(self, input_signal: Tensor, output_signal: Tensor) -> (Tensor, Tensor):
        return reshape(input_signal, (-1, 1)), reshape(output_signal, (-1, 1))


class ResampleSignals(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], original_frequency: int, new_frequency: int):
        super().__init__(out_type)
        self.original_frequency = original_frequency
        self.new_frequency = new_frequency

    def transform(self, input_signal: ndarray, output_signal: ndarray) -> Any:
        return self.resample_signal(input_signal), self.resample_signal(output_signal)

    def resample_signal(self, signal: ndarray) -> ndarray:
        original_samples = signal.shape[0]
        new_samples = int(original_samples * (self.new_frequency / self.original_frequency))
        return resample(signal, new_samples)
