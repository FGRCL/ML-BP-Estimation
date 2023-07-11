from tensorflow import bool, float32, reduce_all
from tensorflow.python.ops.array_ops import size

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal, SqiFiltering
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, MakeWindows, RemoveOutputSignal, SetTensorShape, SignalFilter, \
    StandardizeArray


class WindowPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int = 500, window_size: int = 8, window_step: int = 2, min_pressure: int = 30,
                 max_pressure: int = 230, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8)):
        window_size_frequency = window_size * frequency
        window_step_frequency = window_step * frequency
        super().__init__([
            FilterHasSignal(),
            FilterSize(window_size, frequency),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            MakeWindows(window_size_frequency, window_step_frequency),
            SqiFiltering(0.35, 0.8),
            AddBloodPressureOutput(),
            RemoveOutputSignal(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardizeArray(),
            SetTensorShape([window_size_frequency, 1]),
        ])


class FilterSize(FilterOperation):
    def __init__(self, window_size: int, frequency: int):
        self.window_size = window_size
        self.frequency = frequency

    def filter(self, *signals) -> bool:
        return reduce_all(size(signals) > self.window_size * self.frequency)
