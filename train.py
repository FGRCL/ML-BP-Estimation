import pandas as pd
from keras.metrics import MeanAbsoluteError

from src.metrics.standardeviation import StandardDeviation, AbsoluteError
from src.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing
from src.vitaldb.fetchingstrategy.DatasetApi import DatasetApi
from src.vitaldb.casegenerator import VitalFileOptions, VitalDBGenerator
from src.vitaldb.casesplit import get_splits
from src.models.baseline import build_baseline_model
import src.preprocessing.transforms as transforms
import src.preprocessing.filters as filters

import tensorflow as tf

from tensorflow import keras

from src.vitaldb.fetchingstrategy.VitalFileApi import VitalFileApi

frequency = 500
train_split = 0.7
validate_split = 0.15
validate_test = 0.15
batching = False
epochs = 10

options = VitalFileOptions(
    ['SNUADC/ART'],
    1/frequency
)

train_cases, val_cases, test_cases = get_splits([0.7, 0.15, 0.15])

dataset_train = tf.data.Dataset.from_generator(
    lambda: VitalDBGenerator(options, DatasetApi(), train_cases),
    output_signature=(
        tf.TensorSpec(shape=(None, 1), dtype=tf.float64)
    )
).take(1)

dataset_val = tf.data.Dataset.from_generator(
    lambda: VitalDBGenerator(options, DatasetApi(), val_cases),
    output_signature=(
        tf.TensorSpec(shape=(None, 1), dtype=tf.float64)
    )
).take(1)


def abp_low_pass_graph_adapter(x, f):
    return tf.numpy_function(transforms.abp_low_pass, [x, f], tf.float64)


def extract_clean_windows_graph_adapter(x, f: int, window_size: int, step_size: int):
    return tf.numpy_function(transforms.extract_clean_windows, [x, f, window_size, step_size], tf.float64)


def preprocess_dataset(dataset: tf.data.Dataset):
    dataset = dataset.filter(filters.has_data)
    dataset = dataset.map(transforms.remove_nan)
    dataset = dataset.map(lambda x: abp_low_pass_graph_adapter(x, frequency))
    dataset = dataset.map(lambda x: extract_clean_windows_graph_adapter(x, frequency, 8, 2))
    dataset = dataset.flat_map(transforms.to_tensor)
    dataset = dataset.filter(lambda x: filters.pressure_within_bounds(x, 30, 230))
    dataset = dataset.map(transforms.extract_sbp_dbp_from_abp_window)
    dataset = dataset.map(transforms.scale_array)

    if batching:
        dataset = dataset.map(lambda d, l: (tf.reshape(d, shape=(4000, 1)), l))
        dataset = dataset.batch(20)
    else:
        dataset = dataset.map(lambda d, l: (tf.reshape(d, shape=(1, 4000)), l))

    return dataset


dataset_train = preprocess_dataset(dataset_train)
dataset_val = preprocess_dataset(dataset_val)

model = build_baseline_model()
model.summary()
model.compile(optimizer='Adam', loss=keras.losses.MeanSquaredError(),
              metrics=[
                  MeanAbsoluteError(),
                  StandardDeviation(AbsoluteError())
              ]
              )

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1, profile_batch='500,520')
model.fit(dataset_train, epochs=epochs, callbacks=[tensorboard_callback], validation_data=dataset_val)

model.save('models')
