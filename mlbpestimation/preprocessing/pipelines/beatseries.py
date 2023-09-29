from typing import Any, Tuple, Union

from neurokit2 import ppg_clean, ppg_findpeaks
from numpy import argmax, argmin, asarray, empty, float32, ndarray, zeros
from numpy.random import default_rng
from scipy.signal import butter, correlate, correlation_lags, resample, sosfilt
from tensorflow import DType, Tensor, concat, reduce_all, reduce_max, reduce_min, stack
from tensorflow.python.data import Dataset

from mlbpestimation.preprocessing.base import DatasetOperation, DatasetPreprocessingPipeline, NumpyTransformOperation, TransformOperation
from mlbpestimation.preprocessing.pipelines.beatsequencepreprocessing import FilterBeats
from mlbpestimation.preprocessing.shared.filters import FilterSqi, HasData
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import EnsureShape, FlattenDataset, Reshape, SlidingWindow, StandardScaling


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
                 scale_per_signal: bool,
                 bandpass_input: bool,
                 random_seed: int,
                 subsample: float):
        scaling_axis = None if scale_per_signal else (1, 2)
        super(BeatSeriesPreprocessing, self).__init__([
            FilterHasSignal(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff, bandpass_input),
            SplitHeartbeats((float32, float32), frequency, beat_length),
            FilterBeats(sequence_steps),
            SlidingWindow(sequence_steps, sequence_stride),
            Subsample((float32, float32), random_seed, subsample),
            HasData(),
            FilterSqi((float32, float32), 0.5, 2),
            HasData(),
            AddBloodPressureSequence(),
            EnsureShape([None, sequence_steps, beat_length], [None, sequence_steps, 2]),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=scaling_axis),
            Reshape([-1, sequence_steps, beat_length], [-1, sequence_steps, 2]),
            FlattenDataset(),
            AddShiftedInput(),
            Shuffle(),
        ])


class AddBloodPressureSequence(TransformOperation):
    def transform(self, input_windows: Tensor, output_windows: Tensor) -> Any:
        sbp = reduce_max(output_windows, axis=-1)
        dbp = reduce_min(output_windows, axis=-1)
        pressures = stack((sbp, dbp), -1)

        return input_windows, pressures


class FilterPressureWithinBounds(TransformOperation):
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


class Shuffle(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        count = int(dataset.reduce(0, lambda x, _: x + 1))
        return dataset.shuffle(count)


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


class AdjustPhaseLag(NumpyTransformOperation):
    def transform(self, input_signal: ndarray, output_signal: ndarray) -> (ndarray, ndarray):
        mode = 'full'
        correlation = correlate(input_signal, output_signal, mode=mode)
        lags = correlation_lags(input_signal.shape[0], output_signal.shape[0], mode=mode)
        phase_lag = lags[argmax(correlation)]

        if phase_lag > 0:
            return input_signal[phase_lag:], output_signal[:-phase_lag]
        else:
            return input_signal[:-phase_lag], output_signal[phase_lag:]


class AddShiftedInput(TransformOperation):
    def transform(self, input_windows: Tensor, output_pressures: Tensor) -> Any:
        first = zeros((1, output_pressures.shape[1]))
        shifted_output = concat([first, output_pressures[1:]], axis=0)
        return (input_windows, shifted_output), output_pressures


class Subsample(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], seed: int, sample_rate: float):
        super().__init__(out_type)
        self.random_generator = default_rng(seed)
        self.sample_rate = sample_rate

    def transform(self, inputs_windows: ndarray, output_windows: ndarray) -> Any:
        n_elements = inputs_windows.shape[0]
        sample_size = int(n_elements * self.sample_rate)
        sample_idx = self.random_generator.choice(range(n_elements), sample_size, False)

        print(inputs_windows.shape, output_windows.shape, inputs_windows[sample_idx].shape, output_windows[sample_idx].shape)

        return inputs_windows[sample_idx], output_windows[sample_idx]
