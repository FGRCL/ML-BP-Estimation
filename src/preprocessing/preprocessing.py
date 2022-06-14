import numpy
from tensorflow import Tensor
import tensorflow as tf
import heartpy as hp
import numpy as np


def print_and_return(x: Tensor, y: Tensor = None):
    if y is None:
        tf.print("x: ", x)
        return x
    else:
        tf.print("x: ", x, "y: ", y)
        return x, y


def add_0_labels(x: Tensor):
    return x, 0


def extract_abp_track(tracks: Tensor):
    return tracks[:, 1]


def abp_lowpass(unfiltered_abp: numpy.ndarray, sample_rate: int):
    return hp.filter_signal(unfiltered_abp, cutoff=5, sample_rate=sample_rate, filtertype='lowpass')


def remove_nan(abp_recording: Tensor):
    tf.where(abp_recording).numpy()
    return abp_recording


def abp_split_windows(full_abp_recording: numpy.ndarray, sample_rate: int, window_size: int, step_size: int):
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


def flatten_dataset(x):
    return tf.data.Dataset.from_tensor_slices(x)


def normalize_array(x,y):
    normalized, norm = tf.linalg.normalize(x)
    return normalized, y


def calculate_window_count(size: int, window_size: int, window_step: int) -> int:
    if window_size < size:
        return int(((size - window_size) / window_step)) + 1
    else:
        return 0
