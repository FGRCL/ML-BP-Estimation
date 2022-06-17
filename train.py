from src.vitaldb.fetchingstrategy.Sdk import Sdk
from src.vitaldb.casegenerator import VitalFileOptions
from src.vitaldb.casesplit import split_generator
from src.models.baseline import baseline_model
import src.preprocessing.transforms as transforms
import src.preprocessing.filters as filters

import tensorflow as tf

from tensorflow import keras
from keras import layers

frequency = 500
samples = range(1, 10)
train_split = 0.7
validate_split = 0.15
validate_test = 0.15
batching = True

options = VitalFileOptions(
    ['SNUADC/ART'],
    1/frequency
)

train_generator, val_generator, test_generator = split_generator(options, Sdk(), samples, [0.7, 0.15, 0.15])

dataset_train = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 2), dtype=tf.float64)
    )
)

dataset_val = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 2), dtype=tf.float64)
    )
)

def abp_low_pass_graph_adapter(x, frequency):
    return tf.numpy_function(transforms.abp_low_pass, [x, frequency], tf.float64)

def extract_clean_windows_graph_adapter(x, frequency: int, window_size: int, step_size: int):
    return tf.numpy_function(transforms.extract_clean_windows, [x, frequency, window_size, step_size], tf.float64)

def preprocess_dataset(dataset: tf.data.Dataset):
    dataset = dataset.filter(filters.has_data)
    dataset = dataset.map(transforms.extract_abp_track)
    dataset = dataset.map(transforms.remove_nan)
    dataset = dataset.map(lambda x: abp_low_pass_graph_adapter(x, frequency))
    dataset = dataset.map(lambda x: extract_clean_windows_graph_adapter(x, frequency, 8, 2))
    dataset= dataset.flat_map(transforms.to_tensor)
    dataset = dataset.filter(lambda x: filters.pressure_out_of_bounds(x, 30, 230))
    dataset = dataset.map(transforms.extract_sbp_dbp_from_abp_window)
    dataset = dataset.map(transforms.scale_array)

    if batching:
        dataset = dataset.map(lambda d, l: (tf.reshape(d, shape=(4000, 1)), l))
        dataset = dataset.batch(20).prefetch(2)
    else:
        dataset = dataset.map(lambda d, l: (tf.reshape(d, shape=(1, 4000)), l))

    return dataset

dataset_train = preprocess_dataset(dataset_train)
dataset_val = preprocess_dataset(dataset_val)

model = baseline_model()
model.summary()
model.compile(optimizer='Adam', loss=keras.losses.MeanAbsoluteError())

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="src/logs")
model.fit(dataset_train, epochs=1, callbacks=[tensorboard_callback])

model.evaluate(dataset_val,  callbacks=[tensorboard_callback])
