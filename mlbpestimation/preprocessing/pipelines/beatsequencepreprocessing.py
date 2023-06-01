from typing import Tuple

from tensorflow.python.data import Dataset

from mlbpestimation.preprocessing.base import DatasetOperation, DatasetPreprocessingPipeline
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing


class BeatSequencePreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int, lowpass_cutoff: int, bandpass_cutoff: Tuple[float, float], min_pressure: int, max_pressure: int, beat_length: int,
                 sequence_steps: int,
                 sequence_stride: int):
        super(BeatSequencePreprocessing, self).__init__([
            HeartbeatPreprocessing(
                frequency,
                lowpass_cutoff,
                bandpass_cutoff,
                min_pressure,
                max_pressure,
                beat_length,
            ),
            Window(sequence_steps, sequence_stride)
        ])


class Window(DatasetOperation):
    def __init__(self, size: int, shift: int = None, stride: int = 1):
        self.size = size
        self.shift = shift
        self.stride = stride

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.window(self.size, self.shift, self.stride)
