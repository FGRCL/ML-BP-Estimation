from neurokit2 import ppg_clean, ppg_findpeaks
from numpy import argmin, empty, float32, ndarray, zeros
from tensorflow import DType, Tensor, reduce_all, reduce_max, reduce_min, stack
from tensorflow.python.data import Dataset
from tensorflow.python.ops.array_ops import shape
from typing import Any, Tuple, Union

from mlbpestimation.preprocessing.base import DatasetOperation, DatasetPreprocessingPipeline, FilterOperation, NumpyTransformOperation, TransformOperation
from mlbpestimation.preprocessing.shared.filters import FilterSqi, HasData
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import EnsureShape, FlattenDataset, Reshape, SignalFilter, SlidingWindow, SplitHeartbeats, StandardScaling


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
        scaling_axis = -1 if scale_per_signal else 2
        super(BeatSeriesPreprocessing, self).__init__([
            FilterHasSignal(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            SplitHeartbeats((float32, float32), frequency, beat_length),
            FilterBeats(sequence_steps),
            SlidingWindow(sequence_steps, sequence_stride),
            FilterSqi((float32, float32), 0.5, 2),
            HasData(),
            AddBeatSequenceBloodPressure(),
            EnsureShape([None, sequence_steps, beat_length], [None, sequence_steps, 2]),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=scaling_axis),
            Reshape([-1, sequence_steps, beat_length, 1], [-1, sequence_steps, 2]),
            FlattenDataset(),
            Shuffle(),
        ])


class AddBeatSequenceBloodPressure(TransformOperation):
    def transform(self, input_windows: Tensor, output_windows: Tensor) -> Any:
        sbp = reduce_max(output_windows, axis=-1)
        dbp = reduce_min(output_windows, axis=-1)
        pressures = stack((sbp, dbp), -1)

        return input_windows, pressures


class FilterBeats(FilterOperation):
    def __init__(self, min_beats: int):
        self.min_beats = min_beats

    def filter(self, input_windows: Tensor, output_windows: Tensor) -> bool:
        return shape(input_windows)[0] > self.min_beats


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


class SplitHeartbeatsAlt(NumpyTransformOperation):

    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], frequency, beat_length):
        super().__init__(out_type)
        self.beat_length = beat_length
        self.frequency = frequency

    def transform(self, lowpass_signal: Tensor, bandpass_signal: Tensor = None) -> Any:
        try:
            peak_indices = ppg_findpeaks(ppg_clean(bandpass_signal, self.frequency), self.frequency)['PPG_Peaks']
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
        beats = zeros((beat_count, self.beat_length), dtype=float32)

        for i, (pulse_onset_index, diastolic_foot_index) in enumerate(zip(trough_indices[:-1], trough_indices[1:])):
            beat = signal[pulse_onset_index: diastolic_foot_index]
            beat_truncated = beat[:self.beat_length]
            length = beat_truncated.shape[0]
            beats[i, 0:length] = beat_truncated

        return beats


class Shuffle(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        count = int(dataset.reduce(0, lambda x, _: x + 1))
        return dataset.shuffle(count)
