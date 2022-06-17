from keras.layers import Layer
import tensorflow as tf

from src.preprocessing import filters, transforms


def abp_low_pass_graph_adapter(x, frequency):
    return tf.numpy_function(transforms.abp_low_pass, [x, frequency], tf.float64)


def extract_clean_windows_graph_adapter(x, frequency: int, window_size: int, step_size: int):
    return tf.numpy_function(transforms.extract_clean_windows, [x, frequency, window_size, step_size], tf.float64)


class WindowPreprocessing(Layer):
    def __init__(self, frequency: int, window_size: int, window_step: int):
        super(WindowPreprocessing, self).__init__()
        self.frequency = frequency
        self.window_size = window_size
        self.window_step = window_step

    def call(self, inputs, **kwargs):
        dataset = inputs.filter(filters.has_data)
        dataset = dataset.map(transforms.extract_abp_track)
        dataset = dataset.map(transforms.remove_nan)
        dataset = dataset.map(lambda x: abp_low_pass_graph_adapter(x, self.frequency))
        dataset = dataset.map(
            lambda x: extract_clean_windows_graph_adapter(x, self.frequency, self.window_size, self.window_step))
        dataset = dataset.flat_map(transforms.to_tensor)
        dataset = dataset.filter(lambda x: filters.pressure_out_of_bounds(x, 30, 230))
        dataset = dataset.map(transforms.extract_sbp_dbp_from_abp_window)
        dataset = dataset.map(transforms.scale_array)
        return dataset
