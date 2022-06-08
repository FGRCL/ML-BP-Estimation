import sys

import numpy as np

from src.vital.fetchingstrategy.Sdk import Sdk
from src.vital.vitaldbgenerator import VitalDBGenerator, VitalFileOptions
import tensorflow as tf
import matplotlib.pyplot as plt
import heartpy as hp
import neurokit2 as nk
import src.preprocessing.preprocessing as preprocessing
from tensorflow import keras
from tensorflow.keras import layers

frequency = 500

options = VitalFileOptions(
    ['SNUADC/ART'],
    1/frequency
)
vitalDbGenerator = VitalDBGenerator(options, Sdk(), 10)

dataset = tf.data.Dataset.from_generator(
    lambda: vitalDbGenerator,
    output_signature=(
        tf.TensorSpec(shape=(None, 2), dtype=tf.float64)
    )
)

def abp_lowpass_adapted(x):
    f = preprocessing.abp_lowpass
    f = tf.numpy_function(f, [x, frequency], [tf.float64])
    return f

def abp_split_windows_adapted(x):
    f = preprocessing.abp_split_windows
    f = tf.numpy_function(f, [x, frequency, 8, 2], [tf.float64])
    return f



dataset = dataset.map(preprocessing.extract_abp_track)
# dataset = dataset.map(abp_lowpass_adapted) # this one causes all the data to be nan
dataset = dataset.map(abp_split_windows_adapted)
dataset = dataset.flat_map(preprocessing.flatten_dataset)
dataset = dataset.map(preprocessing.extract_sbp_dbp_from_abp_window)
dataset = dataset.map(lambda d, l: (tf.reshape(d, shape=(1, 4000)), l))


inputs = keras.Input(shape=(4000, 1), batch_size=20)
x = layers.Conv1D(64, 15, activation='relu', input_shape=(4000, 1))(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(4)(x)
x = layers.Dropout(0.1)(x)
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.LSTM(64)(x)
outputs = layers.Dense(2)(x)

fake_data = dataset.from_tensor_slices(np.ones((5000,1,4000)))
fake_data = fake_data.map(lambda x: (x,[2,3]))
fake_data.map(preprocessing.print_and_return)

model = keras.Model(inputs, outputs)
model.summary()
model.compile(optimizer='Adam', loss=keras.losses.MeanAbsoluteError())
model.fit(dataset, epochs=1)
model.evalaute()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
