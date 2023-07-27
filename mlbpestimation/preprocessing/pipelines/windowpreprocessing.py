from typing import Any, List, Optional, Tuple, Union

from numpy import ndarray
from scipy.stats import skew
from tensorflow import DType, Tensor, bool, ensure_shape, float32, reduce_all, reduce_max, reduce_min, reshape
from tensorflow.python.data import Options
from tensorflow.python.data.ops.options import AutotuneAlgorithm, AutotuneOptions, ThreadingOptions
from tensorflow.python.ops.array_ops import size, stack
from tensorflow.python.ops.numpy_ops import shape
from tensorflow.python.ops.signal.shape_ops import frame

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation, NumpyTransformOperation, TransformOperation, WithOptions
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import FlattenDataset, SignalFilter, StandardizeInput


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
            SqiFiltering((float32, float32), 0.35, 0.8),
            AddBloodPressureOutput(),
            EnsureShape([None, window_size_frequency], [None, 2]),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardizeInput(axis=1),
            FlattenDataset(),
            Reshape([window_size_frequency, 1], [2]),
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


class SqiFiltering(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], min: float, max: float):
        super().__init__(out_type)
        self.min = min
        self.max = max

    def transform(self, input_windows: ndarray, output_windows: ndarray) -> Tuple[ndarray, ndarray]:
        skewness = skew(input_windows, axis=1)
        valid_idx = (self.min < skewness) & (skewness < self.max)
        return input_windows[valid_idx], output_windows[valid_idx]


class AddBloodPressureOutput(TransformOperation):
    def transform(self, input_windows: ndarray, output_windows: ndarray) -> Any:
        sbp = reduce_max(output_windows, axis=1)
        dbp = reduce_min(output_windows, axis=1)
        pressures = stack((sbp, dbp), axis=1)

        return input_windows, pressures


class FilterPressureWithinBounds(TransformOperation):
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max

    def transform(self, input_windows: Tensor, pressures: Tensor) -> Tuple[ndarray, ndarray]:
        sbp = pressures[:, 0]
        dbp = pressures[:, 1]
        valid_idx = (self.min < sbp) & (sbp < self.max) & (self.min < dbp) & (dbp < self.max)
        return input_windows[valid_idx], pressures[valid_idx]


class PrintShape(TransformOperation):
    def __init__(self, name: str):
        self.name = name

    def transform(self, *args) -> Any:
        for i, a in enumerate(args):
            print(self.name, i, shape(a))
        return args


class EnsureShape(TransformOperation):
    def __init__(self, *shapes: List[Optional[int]]):
        self.shapes = shapes

    def transform(self, *args: Tensor) -> Tuple[Tensor, ...]:
        return tuple((ensure_shape(tensor, shape) for tensor, shape in zip(args, self.shapes)))


class Reshape(TransformOperation):
    def __init__(self, *shapes: List[Optional[int]]):
        self.shapes = shapes

    def transform(self, *args) -> Any:
        return tuple((reshape(tensor, shape) for tensor, shape in zip(args, self.shapes)))
