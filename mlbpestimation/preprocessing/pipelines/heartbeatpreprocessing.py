from typing import Any, Tuple, Union

from neurokit2 import ppg_clean, ppg_findpeaks
from numpy import append, argmin, empty, float32 as nfloat32, ndarray, zeros
from scipy.signal import find_peaks
from tensorflow import DType, Tensor, float32 as tfloat32

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, NumpyFilterOperation, \
    NumpyTransformOperation
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, FilterSqi, HasData
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, ComputeSqi, FlattenDataset, RemoveLowpassTrack, RemoveNan, \
    RemoveSqi, SetTensorShape, SignalFilter, StandardizeArray


class HeartbeatPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency=500, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8), min_pressure=30, max_pressure=230,
                 beat_length=400, max_peak_count=2):
        dataset_operations = [
            HasData(),
            RemoveNan(),
            SignalFilter((tfloat32, tfloat32), frequency, lowpass_cutoff, bandpass_cutoff),
            SplitHeartbeats((tfloat32, tfloat32), frequency, beat_length),
            HasData(),
            FlattenDataset(),
            ComputeSqi((tfloat32, tfloat32, tfloat32)),
            FilterSqi(1, 2),
            RemoveSqi(),
            AddBloodPressureOutput(),
            RemoveLowpassTrack(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardizeArray(),
            SetTensorShape(beat_length),
        ]
        super().__init__(dataset_operations, debug=False)


class SplitHeartbeats(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], frequency, beat_length):
        super().__init__(out_type)
        self.beat_length = beat_length
        self.frequency = frequency

    def transform(self, lowpass_signal: Tensor, bandpass_signal: Tensor = None) -> Any:
        print(bandpass_signal.shape)
        peak_indices = ppg_findpeaks(ppg_clean(bandpass_signal, self.frequency), self.frequency)['PPG_Peaks']
        # peak_indices = ppg_findpeaks(bandpass_signal, self.frequency)['PPG_Peaks']
        if len(peak_indices) < 3:
            return [zeros(0, nfloat32), zeros(0, nfloat32)]

        trough_indices = self._get_troughs(peak_indices, bandpass_signal)

        lowpass_beats = self._get_beats(trough_indices, lowpass_signal)
        bandpass_beats = self._get_beats(trough_indices, bandpass_signal)

        print(bandpass_beats.shape)
        return [lowpass_beats, bandpass_beats]

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
            beat_padded = self._standardize_heartbeat_length(beat)
            beats[i] = beat_padded

        return beats

    def _standardize_heartbeat_length(self, heartbeat):
        missing_length = self.beat_length - len(heartbeat)
        if missing_length > 0:
            last_element = heartbeat[-1]
            padding = [last_element] * missing_length
            return append(heartbeat, padding)
        else:
            return heartbeat[0:self.beat_length]


class FilterExtraPeaks(NumpyFilterOperation):
    def __init__(self, max_peak_count):
        self.max_peak_count = max_peak_count

    def filter(self, heartbeats: ndarray, y: Tensor = None) -> bool:
        return len(find_peaks(heartbeats)) <= self.max_peak_count
