from typing import Any, Tuple

from heartpy import process_segmentwise
from numpy import asarray, empty, ndarray
from tensorflow import DType, float64

from src.preprocessing.base import DatasetPreprocessingPipeline, NumpyTransformOperation
from src.preprocessing.shared.filters import FilterPressureWithinBounds, HasData
from src.preprocessing.shared.transforms import AddBloodPressureOutput, FlattenDataset, RemoveLowpassTrack, RemoveNan, \
    SetTensorShape, SignalFilter, StandardizeArray


class WindowPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int = 500, window_size: int = 8, window_step: int = 2, min_pressure: int = 30,
                 max_pressure: int = 230, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8)):
        dataset_operations = [
            HasData(),
            RemoveNan(),
            SignalFilter(float64, frequency, lowpass_cutoff, bandpass_cutoff),
            SplitWindows(float64, frequency, window_size, window_step),
            FlattenDataset(),
            AddBloodPressureOutput(),
            RemoveLowpassTrack(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardizeArray(),
            SetTensorShape(frequency * window_size),
        ]
        super().__init__(dataset_operations)


class SplitWindows(NumpyTransformOperation):
    def __init__(self, out_type: DType | Tuple[DType], sample_rate: int, window_size: int, step_size: int, ):
        super().__init__(out_type)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.step_size = step_size

    def transform(self, tracks: ndarray, y: ndarray = None) -> Any:
        track_lowpass, track_bandpass = tracks
        segment_overlap = self.step_size / self.window_size

        try:
            working_data, b = process_segmentwise(track_lowpass, self.sample_rate, segment_width=self.window_size,
                                                  segment_overlap=segment_overlap)
        except (RuntimeWarning, UserWarning):
            pass

        window_indices = []
        for i, (start, end) in enumerate(working_data['segment_indices']):
            if len(working_data['removed_beats'][i]) == 0:
                window_indices.append((start, end))

        window_length = self.window_size * self.sample_rate
        windows = empty(shape=(len(window_indices), 2, window_length))
        for i, (start, end) in enumerate(window_indices):
            window = asarray([track_lowpass[start:end], track_bandpass[start:end]])
            windows[i] = window
        return windows
