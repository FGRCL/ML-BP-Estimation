import tensorflow
from tensorflow.python.data import AUTOTUNE, Dataset
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Conv1D, Dense, Dropout, LSTM, MaxPooling1D
from tensorflow.python.keras.models import Model, Sequential

from mlbpestimation.data.multipartdataset import MultipartDataset
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


class Baseline(Model):
    def __init__(self, frequency=500, window_size=8, batch_size=20):
        super().__init__()
        input_shape = (window_size * frequency, 1)
        self.layer = Sequential([
            InputLayer(input_shape, batch_size),
            Conv1D(64, 15, activation='relu'),
            tensorflow.keras.layers.BatchNormalization(),
            MaxPooling1D(4),
            Dropout(0.1),
            LSTM(64, return_sequences=True),
            LSTM(64),
            Dense(2),
        ])

    def call(self, inputs, training=None, mask=None):
        return self.layer(inputs)


def build_baseline_model(datasets: MultipartDataset, batch_size=20, frequency=500, window_size=8):
    input_shape = (window_size * frequency, 1)

    datasets.train = datasets.train.interleave(
        lambda tensor: Dataset.from_tensors(tensor),
        num_parallel_calls=AUTOTUNE
    )
    datasets.train = datasets.train.shuffle(100)

    # TODO improve readability
    pipeline = WindowPreprocessing(frequency=frequency, window_size=window_size)
    datasets = MultipartDataset(
        *[pipeline.preprocess(dataset).batch(batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE).prefetch(
            AUTOTUNE)
            for dataset in
            datasets])
    model = Sequential([
        InputLayer(input_shape, batch_size),
        Conv1D(64, 15, activation='relu'),
        tensorflow.keras.layers.BatchNormalization(),
        MaxPooling1D(4),
        Dropout(0.1),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(2),
    ])
    return datasets, model
