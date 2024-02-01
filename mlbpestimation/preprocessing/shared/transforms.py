from typing import Any, List, Optional, Tuple, Union

import tensorflow
from neurokit2 import ppg_clean, ppg_findpeaks
from numpy import argmax, argmin, asarray, empty, float32, ndarray, zeros
from numpy.random import default_rng
from scipy.signal import butter, correlate, correlation_lags, resample, sosfilt
from scipy.stats import skew
from tensorflow import DType, Tensor, cast, ensure_shape, reduce_max, reduce_mean, reduce_min, reshape, stack
from tensorflow.python.data import Dataset
from tensorflow.python.ops.array_ops import boolean_mask, gather, shape
from tensorflow.python.ops.gen_math_ops import is_nan, logical_not
from tensorflow.python.ops.math_ops import reduce_std
from tensorflow.python.ops.ragged.ragged_math_ops import reduce_all

from mlbpestimation.preprocessing.base import DatasetOperation, FlatMap, NumpyTransformOperation, TransformOperation


class RemoveNan(TransformOperation):
    def transform(self, input_signal: Tensor, output_signal: Tensor) -> Tuple[Tensor]:
        mask = logical_not(is_nan(input_signal)) & logical_not(is_nan(output_signal))
        return boolean_mask(input_signal, mask), boolean_mask(output_signal, mask)


class StandardScaling(TransformOperation):
    def __init__(self, axis):
        self.axis = axis

    def transform(self, input_window: Tensor, pressures: Tensor) -> (Tensor, Tensor):
        mu = reduce_mean(input_window, self.axis, True)
        sigma = reduce_std(input_window, self.axis, True)
        scaled = (input_window - mu) / sigma
        return scaled, pressures


class SignalFilter(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], sample_rate, lowpass_cutoff, bandpass_cutoff, bandpass_input: bool):
        super().__init__(out_type)
        lowpass_filter = butter(2, lowpass_cutoff, 'lowpass', output='sos', fs=sample_rate)
        bandpass_filter = butter(2, bandpass_cutoff, 'bandpass', output='sos', fs=sample_rate)

        self.input_filter = bandpass_filter if bandpass_input else lowpass_filter
        self.output_filter = lowpass_filter

    def transform(self, input_signal: Tensor, output_signal: Tensor) -> Any:
        input_filtered = asarray(sosfilt(self.input_filter, input_signal), dtype=float32)
        output_filtered = asarray(sosfilt(self.output_filter, output_signal), dtype=float32)

        return input_filtered, output_filtered


class AddBloodPressureOutput(TransformOperation):
    def transform(self, input_windows: Tensor, output_windows: Tensor = None) -> Any:
        sbp = reduce_max(output_windows, axis=-1)
        dbp = reduce_min(output_windows, axis=-1)
        pressures = stack((sbp, dbp), 1)

        return input_windows, pressures


class RemoveOutputSignal(TransformOperation):
    def transform(self, input_window: Tensor, output_window: Tensor, pressures: Tensor) -> Any:
        return input_window, pressures


class FlattenDataset(FlatMap):
    @staticmethod
    def flatten(*args) -> Dataset:
        return Dataset.from_tensor_slices(args)


class Cast(TransformOperation):
    def __init__(self, dtype):
        self.dtype = dtype

    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        return cast(x, self.dtype), cast(y, self.dtype)


class ComputeSqi(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], axis: int = 0):
        super().__init__(out_type)
        self.axis = axis

    def transform(self, input_window: ndarray, output_window: ndarray) -> Any:
        sqi = skew(input_window, self.axis)
        return input_window, output_window, asarray(sqi, dtype=float32)


class RemoveSqi(TransformOperation):
    def transform(self, input_window: ndarray, output_window: ndarray, sqi: ndarray) -> Any:
        return input_window, output_window


class MakeWindows(TransformOperation):
    def __init__(self, window_size, step):
        self.window_size = window_size
        self.step = step

    def transform(self, input_signal: Tensor, output_signal: Tensor) -> Dataset:
        return Dataset.from_tensor_slices((input_signal, output_signal)) \
            .window(self.window_size, self.step, drop_remainder=True) \
            .flat_map(lambda low, high: Dataset.zip((low.batch(self.window_size), high.batch(self.window_size))))


class EnsureShape(TransformOperation):
    def __init__(self, *shapes: List[Optional[int]]):
        self.shapes = shapes

    def transform(self, *args: Tensor) -> Tuple[Tensor, ...]:
        return tuple((ensure_shape(tensor, shape) for tensor, shape in zip(args, self.shapes)))


class Reshape(TransformOperation):
    def __init__(self, *shapes: List[Optional[int]]):
        self.shapes = shapes

    def transform(self, *args) -> Any:
        return tuple((reshape(tensor, shape) for tensor, shape in zip(args, self.shapes)))


class SlidingWindow(TransformOperation):
    def __init__(self, width: int, shift: int):
        self.width = width
        self.shift = shift

    def transform(self, input_signal: Tensor, output_signal: Tensor) -> Tuple[Tensor, Tensor]:
        return self._sliding_window(input_signal), self._sliding_window(output_signal)

    def _sliding_window(self, signal):
        hops = (shape(signal)[0] - self.width + self.shift) // self.shift
        window_idx = tensorflow.range(0, self.width) + self.shift * reshape(tensorflow.range(0, hops), (-1, 1))
        return gather(signal, window_idx)


class SplitHeartbeats(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], frequency, beat_length):
        super().__init__(out_type)
        self.beat_length = beat_length
        self.frequency = frequency

    def transform(self, lowpass_signal: Tensor, bandpass_signal: Tensor = None) -> Any:
        try:
            peak_indices = ppg_findpeaks(ppg_clean(lowpass_signal, self.frequency), self.frequency)['PPG_Peaks']
        except:
            return [zeros(0, float32), zeros(0, float32)]

        if len(peak_indices) < 3:
            return [zeros(0, float32), zeros(0, float32)]

        trough_indices = self._get_troughs(peak_indices, bandpass_signal)

        lowpass_beats = self._get_beats(trough_indices, lowpass_signal)
        bandpass_beats = self._get_beats(trough_indices, bandpass_signal)

        return lowpass_beats, bandpass_beats

    def _get_troughs(self, peak_indices, lowpass_signal):
        trough_count = len(peak_indices) - 1
        trough_indices = empty(trough_count, dtype=int)

        for i, (start_peak_index, end_peak_index) in enumerate(zip(peak_indices[:-1], peak_indices[1:])):
            trough = argmin(lowpass_signal[start_peak_index: end_peak_index])
            trough_index = start_peak_index + trough
            trough_indices[i] = trough_index

        return trough_indices

    def _get_beats(self, trough_indices, signal):
        beat_count = len(trough_indices) - 1
        beats = empty((beat_count, self.beat_length), dtype=float32)

        for i, (pulse_onset_index, diastolic_foot_index) in enumerate(zip(trough_indices[:-1], trough_indices[1:])):
            beat = signal[pulse_onset_index: diastolic_foot_index]
            beat_resampled = resample(beat, self.beat_length)
            beats[i] = beat_resampled

        return beats


class AddBloodPressureSeries(TransformOperation):
    def transform(self, input_windows: Tensor, output_windows: Tensor) -> Any:
        sbp = reduce_max(output_windows, axis=-1)
        dbp = reduce_min(output_windows, axis=-1)
        pressures = stack((sbp, dbp), -1)

        return input_windows, pressures


class FilterPressureSeriesWithinBounds(TransformOperation):
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max

    def transform(self, input_windows: Tensor, pressures: Tensor) -> Tuple[ndarray, ndarray]:
        sbp = pressures[:, :, 0]
        dbp = pressures[:, :, 1]
        sbp_min = reduce_all(self.min < sbp, axis=-1)
        sbp_max = reduce_all(sbp < self.max, axis=-1)
        dbp_min = reduce_all(self.min < dbp, axis=-1)
        dbp_max = reduce_all(dbp < self.max, axis=-1)
        valid_idx = sbp_min & sbp_max & dbp_min & dbp_max
        return input_windows[valid_idx], pressures[valid_idx]


class AdjustPhaseLag(NumpyTransformOperation):
    def transform(self, input_signal: ndarray, output_signal: ndarray) -> (ndarray, ndarray):
        mode = 'full'
        correlation = correlate(input_signal, output_signal, mode=mode)
        lags = correlation_lags(input_signal.shape[0], output_signal.shape[0], mode=mode)
        phase_lag = lags[argmax(correlation)]

        if phase_lag == 0:
            shifted_input_signal, shifted_output_signal = input_signal, output_signal
        elif phase_lag > 0:
            shifted_input_signal, shifted_output_signal = input_signal[phase_lag:], output_signal[:-phase_lag]
        else:
            shifted_input_signal, shifted_output_signal = input_signal[:-phase_lag], output_signal[phase_lag:]

        return shifted_input_signal, shifted_output_signal


class Subsample(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], seed: int, sample_rate: float):
        super().__init__(out_type)
        self.random_generator = default_rng(seed)
        self.sample_rate = sample_rate

    def transform(self, inputs_windows: ndarray, output_windows: ndarray) -> Any:
        n_elements = inputs_windows.shape[0]
        sample_size = int(n_elements * self.sample_rate)
        sample_idx = self.random_generator.choice(range(n_elements), sample_size, False)

        return inputs_windows[sample_idx], output_windows[sample_idx]


class Sample(DatasetOperation):
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def apply(self, dataset: Dataset) -> Dataset:
        count = int(dataset.reduce(0, lambda x, _: x + 1))
        size = round(count * self.sample_rate)
        return dataset.take(size)
