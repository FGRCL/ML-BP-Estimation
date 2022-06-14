import tensorflow as tf
from tensorflow import Tensor


@tf.function
def track_contains_nan(track: Tensor, bp: list) -> bool:
    return not tf.reduce_any(tf.math.is_nan(track))


@tf.function
def bp_contains_nan(track: Tensor, bp: list) -> bool:
    return not tf.reduce_any(tf.math.is_nan(bp))


@tf.function
def pressure_out_of_bounds(track: Tensor, min_pressure: int, max_pressure: int) -> bool:
    return tf.reduce_min(track) > min_pressure and tf.reduce_max(track) < max_pressure


@tf.function
def has_data(track: Tensor) -> bool:
    return tf.size(track) != 0