import tensorflow as tf
from tensorflow import Tensor


@tf.function
def track_contains_nan(track: Tensor, bp: list) -> bool:
    return not tf.reduce_any(tf.math.is_nan(track))


@tf.function
def bp_contains_nan(track: Tensor, bp: list) -> bool:
    return not tf.reduce_any(tf.math.is_nan(bp))