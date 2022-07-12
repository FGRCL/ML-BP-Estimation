import tensorflow as tf
from keras.metrics import MeanAbsoluteError
from tensorflow import keras

from mlbpestimation.metrics.standardeviation import AbsoluteError, StandardDeviation
from mlbpestimation.models.heartbeat import build_heartbeat_cnn_model
from mlbpestimation.vitaldb.casesplit import load_vitaldb_dataset


def main():
    epochs = 10

    train, val, _ = load_vitaldb_dataset()
    (train, val), model = build_heartbeat_cnn_model([train, val])

    model.summary()
    model.compile(optimizer='Adam', loss=keras.losses.MeanSquaredError(),
                  metrics=[
                      MeanAbsoluteError(),
                      StandardDeviation(AbsoluteError())
                  ]
                  )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs", histogram_freq=1, profile_batch='500,520')
    model.fit(train, epochs=epochs, callbacks=[tensorboard_callback], validation_data=val)

    model.save('models')


if __name__ == '__main__':
    main()
