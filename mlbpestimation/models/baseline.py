from keras import Sequential, layers

from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


def build_baseline_model(datasets):
    input_shape = (4000, 1)
    pipeline = WindowPreprocessing()
    preprocessed_datasets = [pipeline.preprocess(dataset).batch(20) for dataset in datasets]
    model = Sequential([
        layers.Conv1D(64, 15, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(4),
        layers.Dropout(0.1),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(2),
    ])
    return preprocessed_datasets, model
