from typing import Any, Tuple

from heartpy import process
from numpy import ndarray, argmin, empty, append, asarray
from scipy.signal import find_peaks
from tensorflow import Tensor, float64, DType

from src.preprocessing.base import DatasetPreprocessingPipeline, NumpyTransformOperation, \
    NumpyFilterOperation
from src.preprocessing.shared.filters import HasData, FilterPressureWithinBounds
from src.preprocessing.shared.transforms import RemoveNan, StandardizeArray, SignalFilter, AddBloodPressureOutput, \
    FlattenDataset, RemoveLowpassTrack, SetTensorShape


class HeartbeatPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency=500, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8), min_pressure=30, max_pressure=230,
                 beat_length=400, max_peak_count=2):
        dataset_operations = [
            HasData(),
            RemoveNan(),
            SignalFilter(float64, frequency, lowpass_cutoff, bandpass_cutoff),
            SplitHeartbeats(float64, frequency, beat_length),
            FlattenDataset(),
            AddBloodPressureOutput(),
            RemoveLowpassTrack(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            FilterExtraPeaks(max_peak_count),
            StandardizeArray(),
            SetTensorShape(beat_length)
        ]
        super().__init__(dataset_operations)


class SplitHeartbeats(NumpyTransformOperation):
    def __init__(self, out_type: DType | Tuple[DType], sample_rate, beat_length):
        super().__init__(out_type)
        self.beat_length = beat_length
        self.sample_rate = sample_rate

    def transform(self, tracks: Tensor, y: Tensor = None) -> Any:
        track_lowpass, track_bandpass = tracks
        working_data, measure = process(track_lowpass, self.sample_rate)

        heartbeats_indices = []
        for start, middle, end in zip(working_data['peaklist'][:-2], working_data['peaklist'][1:-1],
                                      working_data['peaklist'][2:]):
            if not (start in working_data['removed_beats'] or middle in working_data['removed_beats'] or end in
                    working_data['removed_beats']):
                start_beat = start + argmin(track_lowpass[start:middle])
                end_beat = middle + argmin(track_lowpass[middle:end])
                heartbeats_indices.append((start_beat, end_beat))

        heartbeats = empty(shape=(len(heartbeats_indices), 2, self.beat_length))
        for i, indices in enumerate(heartbeats_indices):
            heartbeats[i][0] = self._standardize_heartbeat_length(asarray(track_lowpass[indices[0]:indices[1]]))
            heartbeats[i][1] = self._standardize_heartbeat_length(asarray(track_bandpass[indices[0]:indices[1]]))

        return heartbeats

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