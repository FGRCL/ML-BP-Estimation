from keras.layers import Reshape
from tensorflow import float32, reduce_max, reduce_min
from tensorflow.python.keras.backend import stack

from mlbpestimation.models.basemodel import BloodPressureModel


class Windkessel(BloodPressureModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_1 = self.add_weight(name='alpha_1', shape=(1, 1), dtype=float32)
        self.alpha_2 = self.add_weight(name='alpha_2', shape=(1, 1), dtype=float32)
        self.beta_1 = self.add_weight(name='beta_1', shape=(1, 1), dtype=float32)
        self.beta_2 = self.add_weight(name='beta_2', shape=(1, 1), dtype=float32)
        self.m = self.add_weight(name='m', shape=(1, 1), dtype=float32)

    def set_input_shape(self, dataset_spec):
        pass

    def call(self, inputs, training=None, mask=None):
        signal = Reshape((inputs.shape[1],))(inputs)
        transformed_signal = self.m * ((self.alpha_1 * signal ** 2 + self.beta_1 * signal + 1) / (self.alpha_2 * signal ** 2 + self.beta_2 * signal + 1))

        max = reduce_max(transformed_signal, axis=-1)
        min = reduce_min(transformed_signal, axis=-1)
        output = stack((max, min), 1)
        return output
