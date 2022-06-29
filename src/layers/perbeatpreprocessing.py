import tensorflow as tf
from src.preprocessing.filters import has_data
from src.preprocessing.transforms import extract_abp_track, filter_track, remove_nan

def filter_track_graph_adapter(x, frequency):
    return tf.numpy_function(filter_track, [x, frequency], tf.float64)

class WindowPreprocessing(tf.keras.layers.Layer):
    def __init__(self, frequency: int, window_size: int, window_step: int):
        super(WindowPreprocessing, self).__init__()
        self.frequency = frequency
        self.window_size = window_size
        self.window_step = window_step

    def call(self, inputs, **kwargs):
        dataset = inputs.filter(has_data)
        dataset = dataset.map(extract_abp_track)
        dataset = dataset.map(remove_nan)
        dataset = dataset.map(lambda x: filter_track_graph_adapter(x, self.frequency))
        dataset = data
        return dataset
