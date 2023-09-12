from typing import Any, Tuple, Union

import tensorflow
from neurokit2 import ppg_clean, ppg_findpeaks
from numpy import argmin, asarray, empty, float32, int32, ndarray
from numpy.ma import zeros
from scipy.signal import butter, resample, sosfilt
from scipy.stats import skew
from tensorflow import DType, Tensor, float32 as tfloat32, int32 as tint32, logical_and, reduce_all, reduce_max, reduce_mean, reduce_min, reshape, stack
from tensorflow.python.data import Dataset
from tensorflow.python.ops.array_ops import gather, shape, size
from tensorflow.python.ops.math_ops import reduce_std

from mlbpestimation.preprocessing.base import DatasetOperation, DatasetPreprocessingPipeline, FilterOperation, NumpyTransformOperation, TransformOperation
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import EnsureShape, FlattenDataset, Reshape


class BeatSeriesPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self,
                 frequency: int,
                 lowpass_cutoff: int,
                 bandpass_cutoff: Tuple[float, float],
                 min_pressure: int,
                 max_pressure: int,
                 beat_length: int,
                 sequence_steps: int,
                 sequence_stride: int,
                 scale_per_signal: bool):
        scaling_axis = None if scale_per_signal else (1, 2)
        super(BeatSeriesPreprocessing, self).__init__([
            FilterHasSignal(),
            SignalFilter((tfloat32, tfloat32), frequency, lowpass_cutoff),
            SplitHeartbeatsWithSize((tfloat32, tint32, tfloat32, tint32), frequency, beat_length),
            FilterBeats(sequence_steps),
            SlidingWindow(sequence_steps, sequence_stride),
            FilterSqi((tfloat32, tint32, tfloat32, tint32), 0.5, 2),
            HasDataWithLengths(),
            AddBeatSequenceBloodPressure(),
            EnsureShape([None, sequence_steps, beat_length], [None, sequence_steps], [None, sequence_steps, 2]),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=scaling_axis),
            ResampleAndPad((tfloat32, tfloat32), beat_length),
            Reshape([-1, sequence_steps, beat_length], [-1, sequence_steps, 2]),
            FlattenDataset(),
            Shuffle(),
        ])


class AddBeatSequenceBloodPressure(TransformOperation):
    def transform(self, input_windows: Tensor, input_lengths: Tensor, output_windows: Tensor, output_lengths: Tensor) -> Any:
        sbp = reduce_max(output_windows, axis=-1)
        dbp = reduce_min(output_windows, axis=-1)
        pressures = stack((sbp, dbp), -1)

        return input_windows, input_lengths, pressures


class FilterBeats(FilterOperation):
    def __init__(self, min_beats: int):
        self.min_beats = min_beats

    def filter(self, input_windows: Tensor, input_lengths: Tensor, output_windows: Tensor, output_lengths: Tensor) -> bool:
        return shape(input_windows)[0] > self.min_beats


class FilterPressureWithinBounds(TransformOperation):
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max

    def transform(self, input_windows: Tensor, input_lengths: Tensor, pressures: Tensor) -> Tuple[ndarray, ndarray, ndarray]:
        sbp = pressures[:, :, 0]
        dbp = pressures[:, :, 1]
        sbp_min = reduce_all(self.min < sbp, axis=-1)
        sbp_max = reduce_all(sbp < self.max, axis=-1)
        dbp_min = reduce_all(self.min < dbp, axis=-1)
        dbp_max = reduce_all(dbp < self.max, axis=-1)
        valid_idx = sbp_min & sbp_max & dbp_min & dbp_max
        return input_windows[valid_idx], input_lengths[valid_idx], pressures[valid_idx]


class SplitHeartbeatsWithSize(NumpyTransformOperation):

    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], frequency, beat_length):
        super().__init__(out_type)
        self.beat_length = beat_length
        self.frequency = frequency

    def transform(self, lowpass_signal: Tensor, bandpass_signal: Tensor = None) -> (ndarray, ndarray, ndarray, ndarray):
        try:
            peak_indices = ppg_findpeaks(ppg_clean(bandpass_signal, self.frequency), self.frequency)['PPG_Peaks']
        except:
            return [zeros(0, float32), zeros(0, int32), zeros(0, float32), zeros(0, int32)]

        if len(peak_indices) < 3:
            return [zeros(0, float32), zeros(0, int32), zeros(0, float32), zeros(0, int32)]

        trough_indices = self._get_troughs(peak_indices, bandpass_signal)

        lowpass_beats, lowpass_lengths = self._get_beats(trough_indices, lowpass_signal)
        bandpass_beats, bandpass_lengths = self._get_beats(trough_indices, bandpass_signal)

        return lowpass_beats, lowpass_lengths, bandpass_beats, bandpass_lengths

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
        beats = zeros((beat_count, self.beat_length), dtype=float32)
        lengths = empty(beat_count, dtype=int32)

        for i, (pulse_onset_index, diastolic_foot_index) in enumerate(zip(trough_indices[:-1], trough_indices[1:])):
            beat = signal[pulse_onset_index: diastolic_foot_index]
            beat_resampled = resample(beat, self.beat_length)
            beats[i] = beat_resampled
            lengths[i] = beat.shape[0]

        return beats, lengths


class Shuffle(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        count = int(dataset.reduce(0, lambda x, _: x + 1))
        return dataset.shuffle(count)


class SignalFilter(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], sample_rate, lowpass_cutoff):
        super().__init__(out_type)
        self.lowpass_cutoff = lowpass_cutoff
        self.sample_rate = sample_rate

    def transform(self, input_signal: Tensor, output_signal: Tensor) -> Any:
        lowpass_filter = butter(2, self.lowpass_cutoff, 'lowpass', output='sos', fs=self.sample_rate)
        signal_lowpass = asarray(sosfilt(lowpass_filter, output_signal), dtype=float32)

        return signal_lowpass, signal_lowpass


class HasDataWithLengths(FilterOperation):

    def filter(self, input_signal: Tensor, input_lengths: Tensor, output_signal: Tensor, output_lengths: Tensor) -> bool:
        return logical_and(
            size(input_signal) > 1,
            size(output_signal) > 1,
        )


class ResampleAndPad(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], beat_length):
        super().__init__(out_type)
        self.beat_length = beat_length

    def transform(self, input_beats: Tensor, input_lengths: Tensor, pressures: Tensor) -> Any:
        for i, (series, lengths) in enumerate(zip(input_beats, input_lengths)):
            for j, (beat, length) in enumerate(zip(series, lengths)):
                resampled = resample(beat, length)
                input_beats[i, j, :length] = resampled[:self.beat_length]
                input_beats[i, j, length:] = 0

        return input_beats, pressures


class FilterSqi(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], min: float, max: float):
        super().__init__(out_type)
        self.min = min
        self.max = max

    def transform(self,
                  input_windows: ndarray,
                  input_lengths: ndarray,
                  output_windows: ndarray,
                  output_lengths: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        skewness = skew(input_windows, axis=-1)
        valid_idx = (self.min < skewness) & (skewness < self.max)
        valid_idx = reduce_all(valid_idx, tuple(range(1, valid_idx.ndim)))
        return input_windows[valid_idx], input_lengths[valid_idx], output_windows[valid_idx], output_lengths[valid_idx]


class StandardScaling(TransformOperation):
    def __init__(self, axis):
        self.axis = axis

    def transform(self, input_window: Tensor, input_lengths: Tensor, pressures: Tensor) -> (Tensor, Tensor):
        mu = reduce_mean(input_window, self.axis, True)
        sigma = reduce_std(input_window, self.axis, True)
        scaled = (input_window - mu) / sigma
        return scaled, input_lengths, pressures


class SlidingWindow(TransformOperation):
    def __init__(self, width: int, shift: int):
        self.width = width
        self.shift = shift

    def transform(self, input_signal: Tensor, input_lengths: Tensor, output_signal: Tensor, output_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._sliding_window(input_signal), self._sliding_window(input_lengths), \
               self._sliding_window(output_signal), self._sliding_window(output_lengths)

    def _sliding_window(self, signal):
        hops = (shape(signal)[0] - self.width + self.shift) // self.shift
        window_idx = tensorflow.range(0, self.width) + self.shift * reshape(tensorflow.range(0, hops), (-1, 1))
        return gather(signal, window_idx)
