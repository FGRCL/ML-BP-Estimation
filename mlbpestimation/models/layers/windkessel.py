from keras.engine.base_layer import Layer


class Windkessel(Layer):
    def call(self, inputs, *args, **kwargs):
        parameters, signal = inputs
        alpha_1 = parameters[:, 0]
        alpha_2 = parameters[:, 1]
        beta_1 = parameters[:, 2]
        beta_2 = parameters[:, 3]
        m = parameters[:, 4]

        transformed_signal = m * ((alpha_1 * signal ** 2 + beta_1 * signal + 1) / (alpha_2 * signal ** 2 + beta_2 * signal + 1))
        return transformed_signal
