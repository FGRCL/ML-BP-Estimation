from typing import Any, Tuple, Union

from numpy import asarray, float32, ndarray
from scipy.signal import butter, sosfilt
from scipy.stats import skew
from tensorflow import DType, Tensor, cast, reduce_max, reduce_mean, reduce_min, reshape
from tensorflow.python.data import Dataset
from tensorflow.python.ops.array_ops import boolean_mask
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


class StandardizeInput(TransformOperation):
    def __init__(self, axis=0):
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
    def __init__(self, axis: int = 0):
        self.axis = axis

    def transform(self, input_window: Tensor, output_window: Tensor = None) -> Any:
        sbp = reduce_max(output_window, self.axis)
        dbp = reduce_min(output_window, self.axis)
        return input_window, output_window, [sbp, dbp]


class RemoveOutputSignal(TransformOperation):
    def transform(self, input_window: Tensor, output_window: Tensor, pressures: Tensor) -> Any:
        return input_window, pressures


class FlattenDataset(FlatMap):
    @staticmethod
    def flatten(*args) -> Dataset:
        return Dataset.from_tensor_slices(args)


class SetTensorShape(TransformOperation):
    def __init__(self, shape):
        self.shape = shape

    def transform(self, input_window: Tensor, pressures: Tensor = None) -> Any:
        return reshape(input_window, self.shape), reshape(pressures, [2])


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
