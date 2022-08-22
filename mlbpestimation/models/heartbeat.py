import tensorflow
from tensorflow.python.data import AUTOTUNE, Dataset
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, Dense, Dropout, LSTM, MaxPooling1D

from mlbpestimation.data.multipartdataset import MultipartDataset
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing


def build_heartbeat_cnn_model(datasets, frequency=500):
    input_shape = (400, 1)

    datasets.train = datasets.train.interleave(
        lambda tensor: Dataset.from_tensors(tensor),
        num_parallel_calls=AUTOTUNE
    )
    datasets.train = datasets.train.shuffle(100)

    # TODO improve readability
    pipeline = HeartbeatPreprocessing(frequency=frequency)
    preprocessed_datasets = MultipartDataset(
        *[pipeline.preprocess(dataset).batch(20, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE) for dataset in
          datasets])
    model = Sequential([
        Conv1D(64, 15, activation='relu', input_shape=input_shape),
        tensorflow.keras.layers.BatchNormalization(),
        MaxPooling1D(4),
        Dropout(0.1),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(2),
    ])
    return preprocessed_datasets, model
