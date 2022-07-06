from typing import Any, Tuple

from heartpy import filter_signal, process
from numpy import ndarray, argmin, array
from tensorflow import Tensor, float64, reduce_min, reduce_max, DType
from tensorflow.python.data import Dataset

from src.preprocessing.filters import HasData
from src.preprocessing.pipelines.base import DatasetPreprocessingPipeline, TransformOperation, NumpyTransformOperation, \
    FilterOperation, DatasetOperation
from src.preprocessing.transforms import RemoveNan


class HeartbeatPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency=500, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8), min_pressure=30, max_pressure=230):
        dataset_operations = [
            HasData(),
            RemoveNan(),
            FilterTrack(float64, frequency, lowpass_cutoff, bandpass_cutoff),
            ExtractHeartbeats(float64, frequency),
            FlattenDataset(),
            ExtractBloodPressureFromBeat(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            RemoveLowpassTrack(),
        ]
        super().__init__(dataset_operations, debug=True)


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
        return array([track_lowpass, track_bandpass])


class ExtractHeartbeats(NumpyTransformOperation):
    def __init__(self, out_type: DType | Tuple[DType], sample_rate):
        super().__init__(out_type)
        self.sample_rate = sample_rate

    def transform(self, tracks: ndarray, y: ndarray = None) -> Any:
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

        heartbeats_lowpass = [track_lowpass[i[0]:i[1]] for i in heartbeats_indices]
        heartbeats_bandpass = [track_bandpass[i[0]:i[1]] for i in heartbeats_indices]

        print(heartbeats_lowpass)
        print(heartbeats_bandpass)
        return array([heartbeats_lowpass, heartbeats_bandpass])


class FlattenDataset(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.flat_map(self.element_to_dataset)

    @staticmethod
    def element_to_dataset(x: Tensor, y: Tensor = None) -> tuple[Tensor, Tensor | None]:
        return Dataset.from_tensors(x)


class ExtractBloodPressureFromBeat(TransformOperation):
    def transform(self, tracks: Tensor, y: Tensor = None) -> Any:
        track_lowpass = tracks[0]
        sbp = reduce_max(track_lowpass)
        dbp = reduce_min(track_lowpass)
        return tracks, [sbp, dbp]


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
