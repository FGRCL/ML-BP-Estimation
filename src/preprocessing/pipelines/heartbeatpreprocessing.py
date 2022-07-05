from heartpy import filter_signal, process
from numpy import ndarray, argmin
from tensorflow import Tensor, float64, reduce_min

from src.preprocessing.filters import has_data
from src.preprocessing.pipelines.base import DatasetPreprocessingPipeline, TransformOperation, NumpyTransformOperation, \
    FilterOperation
from src.preprocessing.transforms import remove_nan


class HeartbeatPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self):
        dataset_operations = [
            FilterOperation(has_data),
            TransformOperation(remove_nan),
            NumpyTransformOperation(filter_track, float64),
            NumpyTransformOperation(extract_heartbeats, float64)
        ]
        super().__init__(dataset_operations)


def filter_track(track: ndarray):
    track_lowpass = filter_signal(data=track, cutoff=5, sample_rate=500, filtertype='lowpass')
    track_bandpass = filter_signal(data=track, cutoff=[0.1, 8], sample_rate=500, filtertype='bandpass')
    return (track_lowpass, track_bandpass)


def extract_heartbeats(tracks: ndarray):
    track_lowpass, track_bandpass = tracks
    working_data, measure = process(track_lowpass, 500)

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
    return heartbeats_lowpass, heartbeats_bandpass


def filter_pressure_within_bounds(tracks: Tensor, min_pressure, max_pressure):
    heartbeats_lowpass = tracks[0]
    reduce_min(track) > min_pressure and reduce_max(track) < max_pressure

