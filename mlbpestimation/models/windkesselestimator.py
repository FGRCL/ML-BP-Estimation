from keras.engine.input_layer import InputLayer

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.layers.maxminreduce import MaxMinReduce
from mlbpestimation.models.layers.windkessel import Windkessel
from mlbpestimation.models.snnn import Snnn


class WindkesselEstimator(BloodPressureModel):
    def __init__(self):
        super().__init__()

        self.input_layer = None
        self._snnn = Snnn(2, 512, 5)
        self._windkessel = Windkessel()
        self._maxmin = MaxMinReduce()

    def call(self, inputs, training=None, mask=None):
        parameters = self._snnn(inputs)
        transformed = self._windkessel([parameters, inputs])
        pressures = self._maxmin(transformed)
        return pressures

    def set_input_shape(self, dataset_spec):
        self._snnn.set_input_shape(dataset_spec)
        self.input_layer = InputLayer(dataset_spec[0].shape[1:])
