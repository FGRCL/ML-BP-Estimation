from tensorflow import Tensor, greater, less, logical_and, reduce_all
from tensorflow.python.ops.array_ops import size

from mlbpestimation.preprocessing.base import FilterOperation


class HasData(FilterOperation):

    def filter(self, x: Tensor, y: Tensor = None) -> bool:
        return size(x) != 0


class FilterPressureWithinBounds(FilterOperation):
    def __init__(self, min_pressure, max_pressure):
        self.min_pressure = min_pressure
        self.max_pressure = max_pressure

    def filter(self, bandpass_window: Tensor, blood_pressures: Tensor = None) -> bool:
        sbp, dbp = blood_pressures[0], blood_pressures[1]
        return reduce_all(logical_and(less(sbp, self.max_pressure), greater(dbp, self.min_pressure)))


class FilterSqi(FilterOperation):
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def filter(self, lowpass_window: Tensor, bandpass_window: Tensor, sqi: Tensor) -> bool:
        return reduce_all(logical_and(greater(sqi, self.low_threshold), less(sqi, self.high_threshold)))
