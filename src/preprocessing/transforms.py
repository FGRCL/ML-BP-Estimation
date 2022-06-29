import heartpy as hp
import numpy as np
import tensorflow as tf
from numpy import float64
from tensorflow import Tensor

from src.preprocessing.pipelines.base import TransformOperation


def add_0_labels(x: Tensor):
    return x, 0


def extract_abp_track(tracks: Tensor):
    return tracks[:, 1]


def abp_low_pass(unfiltered_abp: np.ndarray, sample_rate: int) -> float64:
    return hp.filter_signal(unfiltered_abp, cutoff=5, sample_rate=float(sample_rate), filtertype='lowpass')


class RemoveNan(TransformOperation):
    def transform(self, x: Tensor, y: Tensor = None) -> Tensor:
        return tf.boolean_mask(x, tf.logical_not(tf.math.is_nan(x)))


def abp_split_windows(full_abp_recording: Tensor, sample_rate: int, window_size: int, step_size: int):
    window_size_freq = sample_rate * window_size
    step_size_freq = sample_rate * step_size
    window_count = calculate_window_count(full_abp_recording.size, window_size_freq, step_size_freq)

    windows = np.empty((window_count, window_size_freq))

    for i, window in enumerate(windows):
        offset = i * step_size_freq
        windows[i] = full_abp_recording[offset:offset + window_size_freq]

    return windows


def extract_sbp_dbp_from_abp_window(abp_window: Tensor):
    sbp = tf.math.reduce_max(abp_window)
    dbp = tf.math.reduce_min(abp_window)
    return abp_window, [sbp, dbp]


def to_tensor(x):
    return tf.data.Dataset.from_tensor_slices(x)


def standardize_track(x, y):  # Unit test this
    mean = tf.math.reduce_mean(x)
    std = tf.math.reduce_std(x)
    scaled = (x - mean) / std
    return scaled, y


def extract_clean_windows(abp_track: np.ndarray, sample_rate: int, window_size: int, step_size: int):
    segment_overlap = step_size / window_size
    working_data, b = hp.process_segmentwise(abp_track, float(sample_rate), segment_width=float(window_size),
                                             segment_overlap=float(segment_overlap))

    sample_windows = []
    for i, (start, end) in enumerate(working_data['segment_indices']):
        if len(working_data['removed_beats'][i]) == 0:
            sample_windows.append(abp_track[start:end])
    return np.asarray(sample_windows)


def calculate_window_count(size: int, window_size: int, window_step: int) -> int:
    if window_size < size:
        return int(((size - window_size) / window_step)) + 1
    else:
        return 0
