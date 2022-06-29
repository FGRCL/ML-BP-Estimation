import tensorflow as tf
from src.preprocessing.transforms import abp_low_pass, extract_clean_windows, extract_abp_track, remove_nan, to_tensor, extract_sbp_dbp_from_abp_window, standardize_track
from src.preprocessing.filters import has_data, pressure_within_bounds


def abp_low_pass_graph_adapter(x, frequency):
    return tf.numpy_function(abp_low_pass, [x, frequency], tf.float64)


def extract_clean_windows_graph_adapter(x, frequency: int, window_size: int, step_size: int):
    return tf.numpy_function(extract_clean_windows, [x, frequency, window_size, step_size], tf.float64)


def preprocess_window(dataset: tf.data.Dataset, frequency, batching):
    dataset = dataset.filter(has_data)
    dataset = dataset.map(extract_abp_track)
    dataset = dataset.map(remove_nan)
    dataset = dataset.map(lambda x: abp_low_pass_graph_adapter(x, frequency))
    dataset = dataset.map(lambda x: extract_clean_windows_graph_adapter(x, frequency, 8, 2))
    dataset = dataset.flat_map(to_tensor)
    dataset = dataset.filter(lambda x: pressure_within_bounds(x, 30, 230))
    dataset = dataset.map(extract_sbp_dbp_from_abp_window)
    dataset = dataset.map(standardize_track)

    if batching:
        dataset = dataset.map(lambda d, l: (tf.reshape(d, shape=(4000, 1)), l))
        dataset = dataset.batch(20)
    else:
        dataset = dataset.map(lambda d, l: (tf.reshape(d, shape=(1, 4000)), l))

    return dataset
