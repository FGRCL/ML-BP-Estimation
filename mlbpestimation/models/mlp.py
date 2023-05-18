from keras.layers import Dense
from tensorflow.python.keras import Model, Sequential


class MLP(Model):
    def __init__(self, neurons):
        self.layers = Sequential()
        for neuron_count in neurons:
            self.layers.append(Dense(neuron_count))

    def call(self, inputs, training=None, mask=None):
        return self.layers(inputs, training, mask)
