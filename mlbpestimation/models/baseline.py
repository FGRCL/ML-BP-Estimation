import tensorflow
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, Dense, Dropout, LSTM, MaxPooling1D
from tensorflow.python.keras.models import Sequential

from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


def build_baseline_model(datasets, batch_size=20, frequency=500, window_size=8):
    input_shape = (window_size * frequency, 1)
    pipeline = WindowPreprocessing(frequency=frequency, window_size=window_size)
    preprocessed_datasets = [pipeline.preprocess(dataset).batch(batch_size, drop_remainder=True) for dataset in
                             datasets]
    model = Sequential([
        Input(input_shape, batch_size),
        Conv1D(64, 15, activation='relu'),
        tensorflow.keras.layers.BatchNormalization(),
        MaxPooling1D(4),
        Dropout(0.1),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(2),
    ])
    return preprocessed_datasets, model
