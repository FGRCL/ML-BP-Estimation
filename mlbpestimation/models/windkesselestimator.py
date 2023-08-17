from keras import Sequential
from keras.activations import selu
from keras.engine.input_layer import InputLayer
from keras.initializers.initializers import LecunNormal, Zeros
from keras.layers import AlphaDropout, SimpleRNN

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.layers.maxminreduce import MaxMinReduce
from mlbpestimation.models.layers.windkessel import Windkessel


class WindkesselEstimator(BloodPressureModel):
    def __init__(self):
        super().__init__()

        self.input_layer = None
        self._feature_extractor = Sequential()
        for _ in range(2):
            self._feature_extractor.add(
                SimpleRNN(512, activation=selu, kernel_initializer=LecunNormal(), bias_initializer=Zeros(), return_sequences=True)
            )
            self._feature_extractor.add(
                AlphaDropout(.05)
            )
        self._feature_extractor.add(
            SimpleRNN(6, activation=None, kernel_initializer=LecunNormal(), bias_initializer=Zeros(), return_sequences=False)
        )
        self._windkessel = Windkessel()
        self._maxmin = MaxMinReduce()

    def call(self, inputs, training=None, mask=None):
        parameters = self._feature_extractor(inputs)
        transformed = self._windkessel([parameters, inputs])
        pressures = self._maxmin(transformed)
        return pressures

    def set_input_shape(self, dataset_spec):
        self.input_layer = InputLayer(dataset_spec[0].shape[1:])
