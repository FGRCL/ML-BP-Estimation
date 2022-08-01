from collections import namedtuple
from typing import Any, Tuple

import tensorflow as tf
from heartpy import filter_signal
from numpy import ndarray
from tensorflow import DType, Tensor, ensure_shape, reduce_max, reduce_min
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
    def __init__(self, out_type: DType | Tuple[DType], sample_rate, lowpass_cutoff, bandpass_cutoff):
        super().__init__(out_type)
        self.bandpass_cutoff = bandpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.sample_rate = sample_rate

    def transform(self, track: ndarray, y: ndarray = None) -> Any:
        track_lowpass = filter_signal(data=track, cutoff=self.lowpass_cutoff, sample_rate=self.sample_rate,
                                      filtertype='lowpass')
        track_bandpass = filter_signal(data=track, cutoff=self.bandpass_cutoff, sample_rate=self.sample_rate,
                                       filtertype='bandpass')
        FilteredTracks = namedtuple('FilteredTracks', ['lowpass', 'bandpass'])
        return [FilteredTracks(track_lowpass, track_bandpass)]


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
    def element_to_dataset(x: Tensor, y: Tensor = None) -> Dataset:
        return Dataset.from_tensor_slices(x)


class SetTensorShape(TransformOperation):
    def __init__(self, input_length):
        self.input_length = input_length

    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        return ensure_shape(x, self.input_length), ensure_shape(y, 2)
