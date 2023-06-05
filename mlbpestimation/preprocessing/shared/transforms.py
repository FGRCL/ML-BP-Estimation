from typing import Any, Tuple, Union

import tensorflow as tf
from numpy import asarray, float32, ndarray
from scipy.signal import butter, sosfilt
from scipy.stats import skew
from tensorflow import DType, Tensor, cast, reduce_max, reduce_min, reshape
from tensorflow.python.data import Dataset

from mlbpestimation.preprocessing.base import FlatMap, NumpyTransformOperation, TransformOperation


class RemoveNan(TransformOperation):
    def transform(self, x: Tensor, y: Tensor = None) -> Tensor:
        return tf.boolean_mask(x, tf.logical_not(tf.math.is_nan(x)))


class StandardizeArray(TransformOperation):
    def transform(self, bandpass_window: Tensor, pressures: Tensor) -> (Tensor, Tensor):
        mean = tf.math.reduce_mean(bandpass_window)
        std = tf.math.reduce_std(bandpass_window)
        scaled = (bandpass_window - mean) / std
        return scaled, pressures


class SignalFilter(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], sample_rate, lowpass_cutoff, bandpass_cutoff):
        super().__init__(out_type)
        self.bandpass_cutoff = bandpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.sample_rate = sample_rate

    def transform(self, signal: ndarray, y: ndarray = None) -> Any:
        lowpass_filter = butter(2, self.lowpass_cutoff, 'lowpass', output='sos', fs=self.sample_rate)
        bandpass_filter = butter(2, self.bandpass_cutoff, 'bandpass', output='sos', fs=self.sample_rate)
        track_lowpass = asarray(sosfilt(lowpass_filter, signal), dtype=float32)
        track_bandpass = asarray(sosfilt(bandpass_filter, signal), dtype=float32)

        return [track_lowpass, track_bandpass]


class AddBloodPressureOutput(TransformOperation):
    def __init__(self, axis: int = 0):
        self.axis = axis

    def transform(self, lowpass_window: Tensor, bandpass_window: Tensor = None) -> Any:
        sbp = reduce_max(lowpass_window, self.axis)
        dbp = reduce_min(lowpass_window, self.axis)
        return lowpass_window, bandpass_window, [sbp, dbp]


class RemoveLowpassTrack(TransformOperation):
    def transform(self, lowpass_window: Tensor, bandpass_window: Tensor, pressures: Tensor) -> Any:
        return bandpass_window, pressures


class FlattenDataset(FlatMap):
    @staticmethod
    def flatten(*args) -> Dataset:
        return Dataset.from_tensor_slices(args)


class SetTensorShape(TransformOperation):
    def __init__(self, input_length):
        self.input_length = input_length

    def transform(self, bandpass_window: Tensor, pressures: Tensor = None) -> Any:
        return reshape(bandpass_window, [self.input_length, 1]), reshape(pressures, [2])


class Cast(TransformOperation):
    def __init__(self, dtype):
        self.dtype = dtype

    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        return cast(x, self.dtype), cast(y, self.dtype)


class ComputeSqi(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], axis: int = 0):
        super().__init__(out_type)
        self.axis = axis

    def transform(self, window_lowpass: ndarray, window_bandpass: ndarray) -> Any:
        sqi = skew(window_bandpass, self.axis)
        return window_lowpass, window_bandpass, asarray(sqi, dtype=float32)


class RemoveSqi(TransformOperation):
    def transform(self, lowpass_window: ndarray, bandpass_window: ndarray, sqi: ndarray) -> Any:
        return lowpass_window, bandpass_window
