import tensorflow as tf
from tensorflow import Tensor

from src.preprocessing.pipelines.base import FilterOperation


@tf.function
def track_contains_nan(track: Tensor, bp: list) -> bool:
    return not tf.reduce_any(tf.math.is_nan(track))


@tf.function
def bp_contains_nan(track: Tensor, bp: list) -> bool:
    return not tf.reduce_any(tf.math.is_nan(bp))


@tf.function
def pressure_within_bounds(track: Tensor, min_pressure: int, max_pressure: int) -> bool:
    return tf.reduce_min(track) > min_pressure and tf.reduce_max(track) < max_pressure


class HasData(FilterOperation):

    def filter(self, x: Tensor, y: Tensor = None) -> bool:
        return tf.size(x) != 0
