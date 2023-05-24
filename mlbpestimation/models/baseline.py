import tensorflow
from tensorflow.python.keras.layers import Conv1D, Dense, Dropout, LSTM, MaxPooling1D
from tensorflow.python.keras.models import Sequential

from mlbpestimation.models.basemodel import BloodPressureModel


class Baseline(BloodPressureModel):
    def __init__(self):
        super().__init__()
        self._layers = Sequential([
            Conv1D(64, 15, activation='relu'),
            tensorflow.keras.layers.BatchNormalization(),
            MaxPooling1D(4),
            Dropout(0.1),
            LSTM(64, return_sequences=True),
            LSTM(64),
            Dense(2),
        ])

    def call(self, inputs, training=None, mask=None):
        return self._layers(inputs)

    def set_input_shape(self, dataset_spec):
        pass

    def get_config(self):
        return {}
