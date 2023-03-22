from typing import Any, Tuple, Union

from numpy import asarray, float32 as float32, ndarray
from scipy.stats import skew
from tensorflow import DType, Tensor, bool, float32 as tfloat32
from tensorflow.python.data import Dataset
from tensorflow.python.ops.array_ops import size

from mlbpestimation.preprocessing.base import DatasetOperation, DatasetPreprocessingPipeline, FilterOperation, \
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
            SignalFilter(tfloat32, frequency, lowpass_cutoff, bandpass_cutoff),
            # TODO use two ouputs instead of a matrix of two elements
            MakeWindows(window_size, window_step, frequency),
            FlattenWindows(),
            ComputeSqi((tfloat32, tfloat32, tfloat32)),
            # FilterSqi(0), TODO proper filtering
            RemoveSqi(),
            AddBloodPressureOutput(),
            RemoveLowpassTrack(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardizeArray(),
            SetTensorShape(frequency * window_size),
        ]
        super().__init__(dataset_operations, debug=True)


class FilterSize(FilterOperation):
    def __init__(self, window_size: int, frequency: int):
        self.window_size = window_size
        self.frequency = frequency

    def filter(self, signal: Tensor) -> bool:
        return size(signal) > self.window_size * self.frequency


class MakeWindows(DatasetOperation):
    def __init__(self, window_size, step_size, frequency):
        self.window_size = window_size
        self.step_size = step_size
        self.frequency = frequency

    def apply(self, dataset: Dataset) -> Dataset:
        window_size_frequency = self.frequency * self.window_size
        step_size_frequency = self.frequency * self.step_size
        return dataset.window(window_size_frequency, step_size_frequency)


class FlattenWindows(FlatMap):
    def flatten(self, windows: Dataset) -> Union[Dataset, Tuple[Dataset, ...]]:
        return windows


class ComputeSqi(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]]):
        super().__init__(out_type)

    def transform(self, windows: ndarray) -> Any:
        sqi = skew(windows[1])
        return windows[0], windows[1], asarray(sqi, dtype=float32)


class FilterSqi(FilterOperation):
    def __init__(self, threshold):
        self.threshold = threshold

    def filter(self, lowpass_window: ndarray, bandpass_window: ndarray, sqi: ndarray) -> bool:
        return sqi > self.threshold


class RemoveSqi(TransformOperation):
    def transform(self, lowpass_window: ndarray, bandpass_window: ndarray, sqi: ndarray) -> Any:
        return lowpass_window, bandpass_window
