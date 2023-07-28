from typing import Any, List, Optional, Tuple, Union

from numpy import asarray, float32, ndarray
from scipy.signal import butter, sosfilt
from scipy.stats import skew
from tensorflow import DType, Tensor, cast, ensure_shape, reduce_max, reduce_mean, reduce_min, reshape
from tensorflow.python.data import Dataset
from tensorflow.python.ops.array_ops import boolean_mask, stack
from tensorflow.python.ops.gen_math_ops import is_nan
from tensorflow.python.ops.math_ops import reduce_std
from tensorflow.python.ops.numpy_ops import logical_not

from mlbpestimation.preprocessing.base import FlatMap, NumpyTransformOperation, TransformOperation


class RemoveNan(TransformOperation):
    def transform(self, *args) -> Tuple[Tensor]:
        return tuple([self._remove_nan_from_tensor(tensor) for tensor in args])

    @staticmethod
    def _remove_nan_from_tensor(tensor: Tensor) -> Tensor:
        return boolean_mask(tensor, logical_not(is_nan(tensor)))


class StandardScaling(TransformOperation):
    def __init__(self, axis):
        self.axis = axis

    def transform(self, input_window: Tensor, pressures: Tensor) -> (Tensor, Tensor):
        mu = reduce_mean(input_window, self.axis, True)
        sigma = reduce_std(input_window, self.axis, True)
        scaled = (input_window - mu) / sigma
        return scaled, pressures


class SignalFilter(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], sample_rate, lowpass_cutoff, bandpass_cutoff):
        super().__init__(out_type)
        self.bandpass_cutoff = bandpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.sample_rate = sample_rate

    def transform(self, input_signal: Tensor, output_signal: Tensor) -> Any:
        lowpass_filter = butter(2, self.lowpass_cutoff, 'lowpass', output='sos', fs=self.sample_rate)
        bandpass_filter = butter(2, self.bandpass_cutoff, 'bandpass', output='sos', fs=self.sample_rate)
        signal_bandpass = asarray(sosfilt(bandpass_filter, input_signal), dtype=float32)
        signal_lowpass = asarray(sosfilt(lowpass_filter, output_signal), dtype=float32)

        return signal_bandpass, signal_lowpass


class AddBloodPressureOutput(TransformOperation):
    def __init__(self, axis):
        self.axis = axis

    def transform(self, input_windows: Tensor, output_windows: Tensor = None) -> Any:
        sbp = reduce_max(output_windows, self.axis)
        dbp = reduce_min(output_windows, self.axis)
        pressures = stack((sbp, dbp), self.axis)

        return input_windows, pressures


class RemoveOutputSignal(TransformOperation):
    def transform(self, input_window: Tensor, output_window: Tensor, pressures: Tensor) -> Any:
        return input_window, pressures


class FlattenDataset(FlatMap):
    @staticmethod
    def flatten(*args) -> Dataset:
        return Dataset.from_tensor_slices(args)


class Cast(TransformOperation):
    def __init__(self, dtype):
        self.dtype = dtype

    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        return cast(x, self.dtype), cast(y, self.dtype)


class ComputeSqi(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], axis: int = 0):
        super().__init__(out_type)
        self.axis = axis

    def transform(self, input_window: ndarray, output_window: ndarray) -> Any:
        sqi = skew(input_window, self.axis)
        return input_window, output_window, asarray(sqi, dtype=float32)


class RemoveSqi(TransformOperation):
    def transform(self, input_window: ndarray, output_window: ndarray, sqi: ndarray) -> Any:
        return input_window, output_window


class MakeWindows(TransformOperation):
    def __init__(self, window_size, step):
        self.window_size = window_size
        self.step = step

    def transform(self, input_signal: Tensor, output_signal: Tensor) -> Dataset:
        return Dataset.from_tensor_slices((input_signal, output_signal)) \
            .window(self.window_size, self.step, drop_remainder=True) \
            .flat_map(lambda low, high: Dataset.zip((low.batch(self.window_size), high.batch(self.window_size))))


class SqiFiltering(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], min: float, max: float, axis: int):
        super().__init__(out_type)
        self.min = min
        self.max = max
        self.axis = axis

    def transform(self, input_windows: ndarray, output_windows: ndarray) -> Tuple[ndarray, ndarray]:
        skewness = skew(input_windows, axis=self.axis)
        valid_idx = (self.min < skewness) & (skewness < self.max)
        return input_windows[valid_idx], output_windows[valid_idx]


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


class FilterPressureWithinBounds(TransformOperation):
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max

    def transform(self, input_windows: Tensor, pressures: Tensor) -> Tuple[ndarray, ndarray]:
        sbp = pressures[:, 0]
        dbp = pressures[:, 1]
        valid_idx = (self.min < sbp) & (sbp < self.max) & (self.min < dbp) & (dbp < self.max)
        return input_windows[valid_idx], pressures[valid_idx]
