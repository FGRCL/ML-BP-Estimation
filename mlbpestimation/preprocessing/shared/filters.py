from typing import Tuple, Union

from numpy import ndarray
from scipy.stats import skew
from tensorflow import DType, Tensor, logical_and, reduce_all
from tensorflow.python.ops.array_ops import size

from mlbpestimation.preprocessing.base import FilterOperation, NumpyTransformOperation, TransformOperation


class HasData(FilterOperation):

    def filter(self, input_signal: Tensor, output_signal: Tensor) -> bool:
        return logical_and(
            size(input_signal) > 1,
            size(output_signal) > 1,
        )


class FilterSqi(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], min: float, max: float):
        super().__init__(out_type)
        self.min = min
        self.max = max

    def transform(self, input_windows: ndarray, output_windows: ndarray) -> Tuple[ndarray, ndarray]:
        skewness = skew(input_windows, axis=-1)
        valid_idx = (self.min < skewness) & (skewness < self.max)
        valid_idx = reduce_all(valid_idx, tuple(range(1, valid_idx.ndim)))
        return input_windows[valid_idx], output_windows[valid_idx]


class FilterPressureWithinBounds(TransformOperation):
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max

    def transform(self, input_windows: Tensor, pressures: Tensor) -> Tuple[ndarray, ndarray]:
        sbp = pressures[:, 0]
        dbp = pressures[:, 1]
        valid_idx = (self.min < sbp) & (sbp < self.max) & (self.min < dbp) & (dbp < self.max)
        return input_windows[valid_idx], pressures[valid_idx]
