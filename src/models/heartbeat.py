from keras import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Dense

from src.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing


def build_heartbeat_cnn_model(datasets):
    input_shape = (400, 1)
    pipeline = HeartbeatPreprocessing()
    preprocessed_datasets = [pipeline.preprocess(dataset).batch(1) for dataset in datasets]
    model = Sequential([
        Conv1D(64, 15, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(4),
        Dropout(0.1),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(2),
    ])
    return preprocessed_datasets, model
