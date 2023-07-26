from typing import Any, Tuple

from tensorflow import Tensor, bool, float32, reduce_all
from tensorflow.python.data import Dataset
from tensorflow.python.ops.array_ops import size
from tensorflow.python.ops.numpy_ops import stack

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation, Print, TransformOperation
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal, SqiFiltering
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, RemoveOutputSignal, SetTensorShape, SignalFilter, \
    StandardizeInput


class WindowPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int = 500, window_size: int = 8, window_step: int = 2, min_pressure: int = 30,
                 max_pressure: int = 230, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8)):
        window_size_frequency = window_size * frequency
        window_step_frequency = window_step * frequency
        super().__init__([
            FilterHasSignal(),
            FilterSize(window_size_frequency),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            SlidingWindow(window_size_frequency, window_step_frequency),
            Print('sliding window'),
            InnerTransform(min_pressure, max_pressure),
            Print('after transform'),
            StandardizeInput(),
            SetTensorShape([window_size_frequency, 1]),
        ])


class FilterSize(FilterOperation):
    def __init__(self, signal_size: int):
        self.signal_size = signal_size

    def filter(self, *signals) -> bool:
        return reduce_all(size(signals) > self.signal_size)


class SlidingWindow(TransformOperation):
    def __init__(self, size: int, shift: int):
        self.size = size
        self.shift = shift

    def transform(self, input_signal: Tensor, output_signal: Tensor) -> Tuple[Tensor, Tensor]:
        return self._window_signal(input_signal), self._window_signal(output_signal)

    def _window_signal(self, signal: Tensor):
        return stack([signal[i:i + self.size] for i in range(0, size(signal) - self.size + 1, self.shift)])


class InnerTransform(TransformOperation):
    def __init__(self, min_pressure, max_pressure):
        self.pipeline = DatasetPreprocessingPipeline([
            SqiFiltering(0.35, 0.8),
            AddBloodPressureOutput(),
            RemoveOutputSignal(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
        ])

    def transform(self, *args) -> Any:
        dataset = Dataset.from_tensor_slices(args)
        return self.pipeline.apply(dataset)
