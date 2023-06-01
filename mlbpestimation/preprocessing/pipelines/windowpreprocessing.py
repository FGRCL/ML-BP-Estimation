from typing import Any, Tuple, Union

from numpy import float32 as float32
from tensorflow import Tensor, bool
from tensorflow.python.data import Dataset
from tensorflow.python.ops.array_ops import size

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation, \
    FlatMap, TransformOperation
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, FilterSqi, HasData
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, ComputeSqi, RemoveLowpassTrack, \
    RemoveNan, \
    RemoveSqi, SetTensorShape, SignalFilter, StandardizeArray


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
            FilterSqi(0.35, 0.8),
            RemoveSqi(),
            AddBloodPressureOutput(),
            RemoveLowpassTrack(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardizeArray(),
            SetTensorShape(frequency * window_size),
        ]
        super().__init__(dataset_operations)


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
