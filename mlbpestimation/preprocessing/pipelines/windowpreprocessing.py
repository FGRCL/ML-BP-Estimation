from typing import Any, Tuple, Union

from numpy import asarray, float32 as float32, ndarray
from scipy.stats import skew
from tensorflow import DType, Tensor, bool
from tensorflow.python.data import Dataset
from tensorflow.python.ops.array_ops import size

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation, \
    FlatMap, NumpyTransformOperation, \
    TransformOperation
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, HasData
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, RemoveLowpassTrack, \
    RemoveNan, \
    SetTensorShape, SignalFilter, StandardizeArray


class WindowPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int = 500, window_size: int = 8, window_step: int = 2, min_pressure: int = 30,
                 max_pressure: int = 230, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8)):
        dataset_operations = [
            HasData(),
            RemoveNan(),
            FilterSize(window_size, frequency),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            MakeWindows(window_size, window_step, frequency),
            FlattenWindows(),
            BatchWindows(window_size, frequency),
            FlattenWindows(),
            ComputeSqi((float32, float32, float32)),
            FilterSqi(0.9),
            RemoveSqi(),
            AddBloodPressureOutput(),
            RemoveLowpassTrack(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardizeArray(),
            SetTensorShape(frequency * window_size),
        ]
        super().__init__(dataset_operations, debug=False)


class FilterSize(FilterOperation):
    def __init__(self, window_size: int, frequency: int):
        self.window_size = window_size
        self.frequency = frequency

    def filter(self, signal: Tensor) -> bool:
        return size(signal) > self.window_size * self.frequency


class MakeWindows(TransformOperation):
    def __init__(self, window_size, step_size, frequency):
        self.window_size = window_size
        self.step_size = step_size
        self.frequency = frequency

    def transform(self, lowpass_signal: Tensor, highpass_signal: Tensor) -> Tuple[Dataset, Dataset]:
        window_size_frequency = self.frequency * self.window_size
        step_size_frequency = self.frequency * self.step_size
        lowpass_windows = Dataset.from_tensor_slices(lowpass_signal) \
            .window(window_size_frequency, step_size_frequency, drop_remainder=True)
        highpass_windows = Dataset.from_tensor_slices(highpass_signal) \
            .window(window_size_frequency, step_size_frequency, drop_remainder=True)
        return lowpass_windows, highpass_windows


class BatchWindows(TransformOperation):
    def __init__(self, window_size, frequency):
        self.window_size = window_size
        self.frequency = frequency

    def transform(self, lowpass_window_dataset, bandpass_window_dataset) -> Any:
        window_size_frequency = self.window_size * self.frequency
        return lowpass_window_dataset.batch(window_size_frequency), bandpass_window_dataset.batch(window_size_frequency)


class FlattenWindows(FlatMap):
    def flatten(self, lowpass_windows: Dataset, bandpass_windows: Dataset) -> Union[Dataset, Tuple[Dataset, ...]]:
        return Dataset.zip((lowpass_windows, bandpass_windows))


class ComputeSqi(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]]):
        super().__init__(out_type)

    def transform(self, window_lowpass: ndarray, window_bandpass: ndarray) -> Any:
        sqi = skew(window_bandpass)
        return window_lowpass, window_bandpass, asarray(sqi, dtype=float32)


class FilterSqi(FilterOperation):
    def __init__(self, threshold):
        self.threshold = threshold

    def filter(self, lowpass_window: ndarray, bandpass_window: ndarray, sqi: ndarray) -> bool:
        return sqi < self.threshold


class RemoveSqi(TransformOperation):
    def transform(self, lowpass_window: ndarray, bandpass_window: ndarray, sqi: ndarray) -> Any:
        return lowpass_window, bandpass_window
