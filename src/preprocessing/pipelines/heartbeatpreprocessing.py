from collections import namedtuple
from typing import Any, Tuple

from heartpy import filter_signal, process
from numpy import ndarray, argmin, empty, append, asarray
from scipy.signal import find_peaks
from tensorflow import Tensor, float64, reduce_min, reduce_max, DType
from tensorflow.python.data import Dataset

from src.preprocessing.filters import HasData
from src.preprocessing.pipelines.base import DatasetPreprocessingPipeline, TransformOperation, NumpyTransformOperation, \
    FilterOperation, DatasetOperation, NumpyFilterOperation
from src.preprocessing.transforms import RemoveNan, StandardizeArray


class HeartbeatPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency=500, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8), min_pressure=30, max_pressure=230,
                 beat_length=400, max_peak_count=2):
        dataset_operations = [
            HasData(),
            RemoveNan(),
            FilterTrack(float64, frequency, lowpass_cutoff, bandpass_cutoff),
            ExtractHeartbeats(float64, frequency, beat_length),
            FlattenDataset(),
            ExtractBloodPressureFromBeat(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            FilterExtraPeaks(max_peak_count),
            RemoveLowpassTrack(),
            StandardizeArray(),
        ]
        super().__init__(dataset_operations)


class FilterTrack(NumpyTransformOperation):
    def __init__(self, out_type: DType | Tuple[DType], sample_rate, lowpass_cutoff, bandpass_cutoff):
        super().__init__(out_type)
        self.bandpass_cutoff = bandpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.sample_rate = sample_rate

    def transform(self, track: ndarray, y: ndarray = None) -> Any:
        track_lowpass = filter_signal(data=track, cutoff=self.lowpass_cutoff, sample_rate=self.sample_rate,
                                      filtertype='lowpass')
        track_bandpass = filter_signal(data=track, cutoff=self.bandpass_cutoff, sample_rate=self.sample_rate,
                                       filtertype='bandpass')
        FilteredTracks = namedtuple('FilteredTracks', ['lowpass', 'bandpass'])
        return [FilteredTracks(track_lowpass, track_bandpass)]


class ExtractHeartbeats(NumpyTransformOperation):
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


class FlattenDataset(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.flat_map(self.element_to_dataset)

    @staticmethod
    def element_to_dataset(x: Tensor, y: Tensor = None) -> tuple[Tensor, Tensor | None]:
        return Dataset.from_tensor_slices(x)


class ExtractBloodPressureFromBeat(TransformOperation):
    def transform(self, tracks: Tensor, y: Tensor = None) -> Any:
        track_lowpass = tracks[0]
        sbp = reduce_max(track_lowpass)
        dbp = reduce_min(track_lowpass)
        return tracks, [sbp, dbp]


class FilterExtraPeaks(NumpyFilterOperation):
    def __init__(self, max_peak_count):
        self.max_peak_count = max_peak_count

    def filter(self, heartbeats: ndarray, y: Tensor = None) -> bool:
        return len(find_peaks(heartbeats[1])) <= self.max_peak_count


class FilterPressureWithinBounds(FilterOperation):
    def __init__(self, min_pressure, max_pressure):
        self.min_pressure = min_pressure
        self.max_pressure = max_pressure

    def filter(self, tracks: Tensor, y: Tensor = None) -> bool:
        heartbeats_lowpass = tracks[0]
        return reduce_min(heartbeats_lowpass) > self.min_pressure and reduce_max(heartbeats_lowpass) < self.max_pressure


class RemoveLowpassTrack(TransformOperation):
    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        return x[1], y
