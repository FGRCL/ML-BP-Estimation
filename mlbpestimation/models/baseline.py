from keras import Input, Sequential, layers

from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


def build_baseline_model(datasets, batch_size=20, frequency=500, window_size=8):
    input_shape = (window_size * frequency, 1)
    pipeline = WindowPreprocessing(frequency=frequency, window_size=window_size)
    preprocessed_datasets = [pipeline.preprocess(dataset).batch(batch_size) for dataset in datasets]
    model = Sequential([
        Input(input_shape, batch_size),
        layers.Conv1D(64, 15, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(4),
        layers.Dropout(0.1),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(2),
    ])
    return preprocessed_datasets, model
