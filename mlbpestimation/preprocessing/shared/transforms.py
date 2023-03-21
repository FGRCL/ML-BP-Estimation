from typing import Any, Tuple, Union

import tensorflow as tf
from heartpy import filter_signal
from numpy import asarray, float32, ndarray
from tensorflow import DType, Tensor, cast, reduce_max, reduce_min, reshape
from tensorflow.python.data import Dataset

from mlbpestimation.preprocessing.base import DatasetOperation, NumpyTransformOperation, TransformOperation


class RemoveNan(TransformOperation):
    def transform(self, x: Tensor, y: Tensor = None) -> Tensor:
        return tf.boolean_mask(x, tf.logical_not(tf.math.is_nan(x)))


class StandardizeArray(TransformOperation):
    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        mean = tf.math.reduce_mean(x)
        std = tf.math.reduce_std(x)
        scaled = (x - mean) / std
        return scaled, y


class SignalFilter(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], sample_rate, lowpass_cutoff, bandpass_cutoff):
        super().__init__(out_type)
        self.bandpass_cutoff = bandpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.sample_rate = sample_rate

    def transform(self, track: ndarray, y: ndarray = None) -> Any:
        track_lowpass = filter_signal(data=track, cutoff=self.lowpass_cutoff, sample_rate=self.sample_rate,
                                      filtertype='lowpass')
        track_bandpass = filter_signal(data=track, cutoff=self.bandpass_cutoff, sample_rate=self.sample_rate,
                                       filtertype='bandpass')

        filtered_tracks = asarray([track_lowpass, track_bandpass], dtype=float32)
        return filtered_tracks


class AddBloodPressureOutput(TransformOperation):
    def transform(self, tracks: Tensor, y: Tensor = None) -> Any:
        track_lowpass = tracks[0]
        sbp = reduce_max(track_lowpass)
        dbp = reduce_min(track_lowpass)
        return tracks, [sbp, dbp]


class RemoveLowpassTrack(TransformOperation):
    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        return x[1], y


class FlattenDataset(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.flat_map(self.element_to_dataset)

    @staticmethod
    def element_to_dataset(*args) -> Dataset:
        return Dataset.from_tensor_slices(args)


class SetTensorShape(TransformOperation):
    def __init__(self, input_length):
        self.input_length = input_length

    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        return reshape(x, [self.input_length, 1]), reshape(y, [2])


class Cast(TransformOperation):
    def __init__(self, dtype):
        self.dtype = dtype

    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        return cast(x, self.dtype), cast(y, self.dtype)
