from keras import Sequential, layers


def build_baseline_model():
    input_shape = (4000, 1)
    return Sequential([
        layers.Conv1D(64, 15, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(4),
        layers.Dropout(0.1),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(2),
    ])
