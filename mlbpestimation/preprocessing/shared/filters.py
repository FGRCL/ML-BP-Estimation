import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.ops.array_ops import size

from mlbpestimation.preprocessing.base import FilterOperation


class HasData(FilterOperation):

    def filter(self, x: Tensor, y: Tensor = None) -> bool:
        return tf.size(x) != 0


class FilterPressureWithinBounds(FilterOperation):
    def __init__(self, min_pressure, max_pressure):
        self.min_pressure = min_pressure
        self.max_pressure = max_pressure

    def filter(self, tracks: Tensor, blood_pressures: Tensor = None) -> bool:
        sbp, dbp = blood_pressures[0], blood_pressures[1]
        return sbp < self.max_pressure and dbp > self.min_pressure


class HasFrames(FilterOperation):
    def filter(self, x: Tensor, y: Tensor = None) -> bool:
        return size(x) > 1
