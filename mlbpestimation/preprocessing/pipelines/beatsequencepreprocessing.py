from typing import Any, Tuple, Union

from numpy import ndarray
from tensorflow import Tensor, float32, greater, less, logical_and, reduce_all
from tensorflow.python.data import Dataset

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation, FlatMap, TransformOperation
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import SplitHeartbeats
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import MakeWindows
from mlbpestimation.preprocessing.shared.filters import HasData
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, ComputeSqi, RemoveLowpassTrack, RemoveNan, RemoveSqi, SignalFilter


class BeatSequencePreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int, lowpass_cutoff: int, bandpass_cutoff: Tuple[float, float], min_pressure: int, max_pressure: int, beat_length: int,
                 sequence_steps: int,
                 sequence_stride: int):
        super(BeatSequencePreprocessing, self).__init__([
            HasData(),
            RemoveNan(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            SplitHeartbeats((float32, float32), frequency, beat_length),
            HasData(),
            MakeWindows(sequence_steps, sequence_stride),
            FlattenWindows(),
            BatchWindows(sequence_steps),
            FlattenWindows(),
            ComputeSqi((float32, float32, float32), 1),
            FilterSqiList(0.5, 2),
            RemoveSqi(),
            AddBloodPressureOutput(1),
            FilterPressureWithinBoundsList(min_pressure, max_pressure),
            RemoveLowpassTrack(),
        ])


class MakeWindows(TransformOperation):
    def __init__(self, window_size, step):
        self.window_size = window_size
        self.step = step

    def transform(self, lowpass_beat: Tensor, highpass_beat: Tensor) -> Tuple[Dataset, Dataset]:
        lowpass_windows = Dataset.from_tensor_slices(lowpass_beat) \
            .window(self.window_size, self.step, drop_remainder=True)
        highpass_windows = Dataset.from_tensor_slices((lowpass_beat, highpass_beat)) \
            .window(self.window_size, self.step, drop_remainder=True)
        return lowpass_windows, highpass_windows


class BatchWindows(TransformOperation):
    def __init__(self, window_size):
        self.window_size = window_size

    def transform(self, lowpass_window_dataset, bandpass_window_dataset) -> Any:
        return lowpass_window_dataset.batch(self.window_size), bandpass_window_dataset.batch(self.window_size)


class FlattenWindows(FlatMap):
    def flatten(self, lowpass_windows: Dataset, bandpass_windows: Dataset) -> Union[Dataset, Tuple[Dataset, ...]]:
        return Dataset.zip((lowpass_windows, bandpass_windows))


class FilterSqiList(FilterOperation):
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def filter(self, lowpass_window: ndarray, bandpass_window: ndarray, sqi: ndarray) -> bool:
        return reduce_all(logical_and(greater(sqi, self.low_threshold), less(sqi, self.high_threshold)))


class FilterPressureWithinBoundsList(FilterOperation):
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def filter(self, lowpass_window: ndarray, bandpass_window: ndarray, sqi: ndarray) -> bool:
        return reduce_all(logical_and(greater(sqi, self.low_threshold), less(sqi, self.high_threshold)))
