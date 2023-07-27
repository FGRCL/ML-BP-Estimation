from typing import Any, Tuple

from numpy import ndarray
from scipy.stats import skew
from tensorflow import Tensor, bool, float32, reduce_all, reduce_max, reduce_min
from tensorflow.python.data import Dataset, Options
from tensorflow.python.data.ops.options import AutotuneAlgorithm, AutotuneOptions, ThreadingOptions
from tensorflow.python.ops.array_ops import size, stack
from tensorflow.python.ops.signal.shape_ops import frame

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation, NumpyFilterOperation, TransformOperation, WithOptions
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import SetTensorShape, SignalFilter, StandardizeInput


class WindowPreprocessing(DatasetPreprocessingPipeline):
    autotune_options = AutotuneOptions()
    autotune_options.autotune_algorithm = AutotuneAlgorithm.MAX_PARALLELISM
    autotune_options.enabled = True
    autotune_options.ram_budget = int(3.2e10)

    threading_options = ThreadingOptions()
    threading_options.private_threadpool_size = 0

    options = Options()
    options.autotune = autotune_options
    options.deterministic = True
    options.threading = threading_options

    def __init__(self, frequency: int = 500, window_size: int = 8, window_step: int = 2, min_pressure: int = 30,
                 max_pressure: int = 230, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8)):
        window_size_frequency = window_size * frequency
        window_step_frequency = window_step * frequency
        super().__init__([
            FilterHasSignal(),
            FilterSize(window_size_frequency),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            SlidingWindow(window_size_frequency, window_step_frequency),
            SqiFiltering(0.35, 0.8),
            AddBloodPressureOutput(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardizeInput(axis=1),
            SetTensorShape([window_size_frequency, 1]),
            WithOptions(self.options)
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
        return frame(input_signal, self.size, self.shift), frame(output_signal, self.size, self.shift)


class InnerTransform(TransformOperation):
    def __init__(self, min_pressure, max_pressure):
        self.pipeline = DatasetPreprocessingPipeline([

        ])

    def transform(self, input_windows: Tensor, output_window: Tensor) -> Any:
        print(input_windows, output_window)
        dataset = Dataset.from_tensor_slices((input_windows, output_window))
        dataset = self.pipeline.apply(dataset)
        for e in dataset:
            print(e)
        return next(iter(dataset.batch(dataset.cardinality())))


class StandardScaling(TransformOperation):
    def transform(self, windows) -> Any:
        print(size(windows))


class SqiFiltering(NumpyFilterOperation):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def filter(self, input_windows: ndarray, output_windows: ndarray) -> Tuple[ndarray, ndarray]:
        accepted_idx = self.min < skew(input_windows) < self.max
        return input_windows[accepted_idx], output_windows[accepted_idx]


class AddBloodPressureOutput(TransformOperation):
    def transform(self, input_windows: ndarray, output_windows: ndarray) -> Any:
        sbp = reduce_max(output_windows, axis=1)
        dbp = reduce_min(output_windows, axis=1)
        pressures = stack((sbp, dbp), axis=1)

        return input_windows, pressures


class FilterPressureWithinBounds(FilterOperation):
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max

    def filter(self, input_windows: Tensor, pressures: Tensor):
        accepted_idx = self.min < pressures[:, 0] < self.max and self.min < pressures[:, 1] < self.max
        return input_windows[accepted_idx], pressures[accepted_idx]
