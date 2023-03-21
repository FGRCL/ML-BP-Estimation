from typing import Any, Tuple, Union

from heartpy import process_segmentwise
from numpy import asarray, empty, ndarray
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import skew
from tensorflow import DType, Tensor, bool, float32

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation, NumpyTransformOperation, \
    TransformOperation
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, HasData
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, FlattenDataset, \
    RemoveLowpassTrack, \
    RemoveNan, \
    SetTensorShape, SignalFilter, StandardizeArray


class WindowPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int = 500, window_size: int = 8, window_step: int = 2, min_pressure: int = 30,
                 max_pressure: int = 230, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8)):
        dataset_operations = [
            HasData(),
            RemoveNan(),
            SignalFilter(float32, frequency, lowpass_cutoff, bandpass_cutoff),
            MakeWindowsIndices((float32, float32), frequency, window_size, window_step),
            FlattenDataset(),
            ComputeSqi((float32, float32, float32)),
            FilterSqi(0),
            RemoveSqi(),
            # SplitWindows(float32, frequency, window_size, window_step),
            # HasData(),
            # FlattenDataset(),
            AddBloodPressureOutput(),
            RemoveLowpassTrack(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardizeArray(),
            SetTensorShape(frequency * window_size),
        ]
        super().__init__(dataset_operations, debug=True)


class SplitWindows(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], sample_rate: int, window_size: int, step_size: int, ):
        super().__init__(out_type)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.step_size = step_size

    def transform(self, tracks: ndarray, y: ndarray = None) -> Any:
        track_lowpass, track_bandpass = tracks
        segment_overlap = self.step_size / self.window_size

        print(f'segment overlap: {segment_overlap}')
        # try:
        working_data, b = process_segmentwise(track_lowpass, self.sample_rate, segment_width=self.window_size,
                                              segment_overlap=segment_overlap,
                                              segment_min_size=self.window_size)
        # except (RuntimeWarning, UserWarning):
        #     pass
        print("working data")
        window_indices = self.get_clean_window_indices(working_data)
        print('window_indices')
        window_length = self.window_size * self.sample_rate
        windows = self.get_windows_from_indices(track_bandpass, track_lowpass, window_indices, window_length)
        return windows

    @staticmethod
    def get_clean_window_indices(working_data):
        window_indices = []
        if 'segment_indices' in working_data:
            for i, (start, end) in enumerate(working_data['segment_indices']):
                if len(working_data['removed_beats'][i]) == 0:
                    window_indices.append((start, end))
        return window_indices

    @staticmethod
    def get_windows_from_indices(track_bandpass, track_lowpass, window_indices, window_length):
        windows = empty(shape=(len(window_indices), 2, window_length))
        for i, (start, end) in enumerate(window_indices):
            window = asarray([track_lowpass[start:end], track_bandpass[start:end]])
            windows[i] = window
        return windows


class MakeWindowsIndices(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], sample_rate, window_size, step_size):
        super().__init__(out_type)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.step_size = step_size

    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        track_lowpass, track_bandpass = x
        print(track_lowpass)
        window_size_frequency = self.sample_rate * self.window_size
        step_size_frequency = self.step_size * self.window_size
        lowpass_windows = sliding_window_view(track_lowpass, window_size_frequency)[:, ::step_size_frequency]
        bandpass_windows = sliding_window_view(track_bandpass, window_size_frequency)[:, ::step_size_frequency]
        return [bandpass_windows, lowpass_windows]


class ComputeSqi(NumpyTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]]):
        super().__init__(out_type)

    def transform(self, bandpass_window: ndarray, lowpass_window: ndarray) -> Any:
        sqi = skew(bandpass_window)
        return [bandpass_window, lowpass_window, asarray(sqi)]


class FilterSqi(FilterOperation):
    def __init__(self, threshold):
        self.threshold = threshold

    def filter(self, bandpass_window: ndarray, lowpass_window: ndarray, sqi: ndarray) -> bool:
        return sqi > self.threshold


class RemoveSqi(TransformOperation):
    def transform(self, bandpass_window: ndarray, lowpass_window: ndarray, sqi: ndarray) -> Any:
        return bandpass_window, lowpass_window
