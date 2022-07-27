import tensorflow as tf
from keras.metrics import MeanAbsoluteError
from tensorflow import keras

from src.data.mimic4.dataset import load_mimic_dataset
from src.metrics.standardeviation import AbsoluteError, StandardDeviation
from src.models.baseline import build_baseline_model

epochs = 10

train, val, _ = load_mimic_dataset()
(train, val), model = build_baseline_model([train, val], frequency=500)

model.summary()
model.compile(optimizer='Adam', loss=keras.losses.MeanSquaredError(),
              metrics=[
                  MeanAbsoluteError(),
                  StandardDeviation(AbsoluteError())
              ]
              )
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1, profile_batch='500,520')
model.fit(train, epochs=epochs, callbacks=[tensorboard_callback], validation_data=val, batch_size=20)

model.save('models')
