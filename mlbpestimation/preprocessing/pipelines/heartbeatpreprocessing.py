from typing import Any, Tuple, Union

from heartpy import process
from heartpy.exceptions import BadSignalWarning
from numpy import append, argmin, asarray, empty, float32 as nfloat32, ndarray
from scipy.signal import find_peaks
from tensorflow import DType, Tensor, float32 as tfloat32

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, NumpyFilterOperation, \
    NumpyTransformOperation
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, HasData
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, FlattenDataset, RemoveLowpassTrack, \
    RemoveNan, \
    SetTensorShape, SignalFilter, StandardizeArray


class HeartbeatPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency=500, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8), min_pressure=30, max_pressure=230,
                 beat_length=400, max_peak_count=2):
        dataset_operations = [
            HasData(),
            RemoveNan(),
            SignalFilter(tfloat32, frequency, lowpass_cutoff, bandpass_cutoff),
            SplitHeartbeats(tfloat32, frequency, beat_length),
            HasData(),
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
    def __init__(self, out_type: Union[DType, Tuple[DType]], sample_rate, beat_length):
        super().__init__(out_type)
        self.beat_length = beat_length
        self.sample_rate = sample_rate

    def transform(self, tracks: Tensor, y: Tensor = None) -> Any:
        track_lowpass, track_bandpass = tracks

        try:
            working_data, measure = process(track_lowpass, self.sample_rate)
        except BadSignalWarning:
            return []

        heartbeats_indices = self._get_clean_heartbeat_indices(track_lowpass, working_data)

        heartbeats = self._get_heartbeat_frames(heartbeats_indices, track_bandpass, track_lowpass)
        heartbeats = asarray(heartbeats, dtype=nfloat32)

        return heartbeats

    def _get_heartbeat_frames(self, heartbeats_indices, track_bandpass, track_lowpass):
        heartbeats = empty(shape=(len(heartbeats_indices), 2, self.beat_length))
        for i, indices in enumerate(heartbeats_indices):
            heartbeats[i][0] = self._standardize_heartbeat_length(asarray(track_lowpass[indices[0]:indices[1]]))
            heartbeats[i][1] = self._standardize_heartbeat_length(asarray(track_bandpass[indices[0]:indices[1]]))
        return heartbeats

    def _get_clean_heartbeat_indices(self, track_lowpass, working_data):
        heartbeats_indices = []
        for start, middle, end in zip(working_data['peaklist'][:-2], working_data['peaklist'][1:-1],
                                      working_data['peaklist'][2:]):
            if not (start in working_data['removed_beats'] or middle in working_data['removed_beats'] or end in
                    working_data['removed_beats']):
                start_beat = start + argmin(track_lowpass[start:middle])
                end_beat = middle + argmin(track_lowpass[middle:end])
                heartbeats_indices.append((start_beat, end_beat))
        return heartbeats_indices

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
