from typing import Any, Tuple, Union

from neurokit2 import ppg_clean, ppg_findpeaks
from numpy import argmin, empty, float32 as nfloat32, ndarray, zeros
from scipy.signal import find_peaks, resample
from tensorflow import DType, Tensor, float32 as tfloat32
from tensorflow.python.data.ops.options import AutotuneAlgorithm, AutotuneOptions, Options, ThreadingOptions

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, NumpyFilterOperation, NumpyTransformOperation, WithOptions
from mlbpestimation.preprocessing.shared.filters import HasData
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, EnsureShape, FilterPressureWithinBounds, FlattenDataset, Reshape, SignalFilter, \
    SqiFiltering, StandardScaling


class HeartbeatPreprocessing(DatasetPreprocessingPipeline):
    autotune_options = AutotuneOptions()
    autotune_options.autotune_algorithm = AutotuneAlgorithm.MAX_PARALLELISM
    autotune_options.enabled = True
    autotune_options.ram_budget = int(3.2e10)

    threading_options = ThreadingOptions()
    threading_options.private_threadpool_size = 0

    options = Options()
    options.autotune = autotune_options
    options.deterministic = True
    options.threading = threading_options

    def __init__(self, frequency=500, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8), min_pressure=30, max_pressure=230,
                 beat_length=400):
        dataset_operations = [
            FilterHasSignal(),
            SignalFilter((tfloat32, tfloat32), frequency, lowpass_cutoff, bandpass_cutoff),
            SplitHeartbeats((tfloat32, tfloat32), frequency, beat_length),
            HasData(),
            SqiFiltering((tfloat32, tfloat32), 0.5, 2, axis=1),
            AddBloodPressureOutput(axis=1),
            EnsureShape([None, beat_length], [None, 2]),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=1),
            Reshape([-1, beat_length, 1], [-1, 2]),
            FlattenDataset(),
            WithOptions(self.options)
        ]
        super().__init__(dataset_operations)


class SplitHeartbeats(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], frequency, beat_length):
        super().__init__(out_type)
        self.beat_length = beat_length
        self.frequency = frequency

    def transform(self, lowpass_signal: Tensor, bandpass_signal: Tensor = None) -> Any:
        try:
            peak_indices = ppg_findpeaks(ppg_clean(bandpass_signal, self.frequency), self.frequency)['PPG_Peaks']
        except:
            return [zeros(0, nfloat32), zeros(0, nfloat32)]

        if len(peak_indices) < 3:
            return [zeros(0, nfloat32), zeros(0, nfloat32)]

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
        beats = empty((beat_count, self.beat_length), dtype=nfloat32)

        for i, (pulse_onset_index, diastolic_foot_index) in enumerate(zip(trough_indices[:-1], trough_indices[1:])):
            beat = signal[pulse_onset_index: diastolic_foot_index]
            beat_resampled = resample(beat, self.beat_length)
            beats[i] = beat_resampled

        return beats


class FilterExtraPeaks(NumpyFilterOperation):
    def __init__(self, max_peak_count):
        self.max_peak_count = max_peak_count

    def filter(self, heartbeats: ndarray, y: Tensor = None) -> bool:
        return len(find_peaks(heartbeats)) <= self.max_peak_count
