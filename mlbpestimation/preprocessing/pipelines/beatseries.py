from itertools import compress
from typing import Tuple, Union

from neurokit2 import ppg_clean, ppg_findpeaks
from numpy import argmin, empty, float32, percentile, where, zeros
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample
from tensorflow import DType, Tensor
from tensorflow.python.ops.ragged.ragged_concat_ops import stack

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, Prefetch, PrintShape, PythonFunctionTransformOperation, Shuffle
from mlbpestimation.preprocessing.pipelines.beatsequencepreprocessing import FilterBeats
from mlbpestimation.preprocessing.shared.filters import FilterSqi, HasData
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureSeries, AdjustPhaseLag, EnsureShape, FilterPressureSeriesWithinBounds, FlattenDataset, Reshape, SignalFilter, SlidingWindow, StandardScaling, Subsample


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
            AdjustPhaseLag((float32, float32)),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff, bandpass_input),
            SegmentHeartbeats((float32, float32), frequency, beat_length, 99.9),
            FlattenDataset(),
            PrintShape(),
            FilterBeats(sequence_steps),
            SlidingWindow(sequence_steps, sequence_stride),
            Subsample((float32, float32), random_seed, subsample),
            HasData(),
            FilterSqi((float32, float32), 0.5, 2),
            HasData(),
            AddBloodPressureSeries(),
            EnsureShape([None, sequence_steps, beat_length], [None, sequence_steps, 2]),
            FilterPressureSeriesWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=scaling_axis),
            Reshape([-1, sequence_steps, beat_length], [-1, sequence_steps, 2]),
            FlattenDataset(),
            Shuffle(),
            Prefetch(),
        ])


class SegmentHeartbeats(PythonFunctionTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], frequency, beat_length, percentile):
        super().__init__(out_type)
        self.beat_length = beat_length
        self.frequency = frequency
        self.percentile = percentile

    def transform(self, lowpass_signal: Tensor, bandpass_signal: Tensor = None):
        try:
            peak_indices = ppg_findpeaks(ppg_clean(lowpass_signal, self.frequency), self.frequency)['PPG_Peaks']
        except:
            return [zeros(0, float32), zeros(0, float32)]

        if len(peak_indices) < 3:
            return [zeros(0, float32), zeros(0, float32)]

        segments_indices = self._get_segments_indices(peak_indices)
        lowpass_segments = []
        bandpass_segments = []
        for segment_indices in segments_indices:
            trough_indices = self._get_troughs(segment_indices, bandpass_signal)

            lowpass_segments.append(self._get_beats(trough_indices, lowpass_signal))
            bandpass_segments.append(self._get_beats(trough_indices, bandpass_signal))

        lowpass_segments = stack(lowpass_segments)
        bandpass_segments = stack(bandpass_segments)
        print(lowpass_segments, bandpass_segments)
        return lowpass_segments, bandpass_segments

    def _get_segments_indices(self, peak_indices):
        intervals = peak_indices[1:] - peak_indices[:-1]
        intervals_diff = abs(gaussian_filter1d(intervals, 5, order=1))
        cutoff = percentile(intervals_diff, [self.percentile])[0]
        segment_ends = where(intervals_diff > cutoff)[0]

        segments = []
        start_index = 0
        for end_index in segment_ends:
            segments.append(peak_indices[start_index: end_index + 1])
            start_index = end_index + 1
        segments.append(peak_indices[start_index:])
        segments = list(compress(segments, [s.shape[0] > 3 for s in segments]))
        return segments

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
