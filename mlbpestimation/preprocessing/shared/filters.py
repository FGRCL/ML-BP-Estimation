from tensorflow import Tensor, greater, less, logical_and, reduce_all
from tensorflow.python.ops.array_ops import size

from mlbpestimation.preprocessing.base import FilterOperation


class HasData(FilterOperation):

    def filter(self, input_signal: Tensor, output_signal: Tensor) -> bool:
        return logical_and(
            size(input_signal) > 1,
            size(output_signal) > 1,
        )


class FilterSqi(FilterOperation):
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def filter(self, input_window: Tensor, output_window: Tensor, sqi: Tensor) -> bool:
        return reduce_all(logical_and(greater(sqi, self.low_threshold), less(sqi, self.high_threshold)))
