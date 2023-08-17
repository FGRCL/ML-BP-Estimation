from keras.engine.base_layer import Layer
from keras.layers import Reshape
from tensorflow.python.ops.array_ops import expand_dims


class Windkessel(Layer):
    def call(self, inputs, *args, **kwargs):
        parameters, signal = inputs
        alpha_1 = expand_dims(parameters[:, 0], -1)
        alpha_2 = expand_dims(parameters[:, 1], -1)
        beta_1 = expand_dims(parameters[:, 2], -1)
        beta_2 = expand_dims(parameters[:, 3], -1)
        m = expand_dims(parameters[:, 4], -1)

        signal = Reshape((signal.shape[1],))(signal)
        transformed_signal = m * ((alpha_1 * signal ** 2 + beta_1 * signal + 1) / (alpha_2 * signal ** 2 + beta_2 * signal + 1))
        return transformed_signal
