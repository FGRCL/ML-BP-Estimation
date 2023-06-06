from typing import Tuple

from tensorflow import Tensor, float32
from tensorflow.python.data import Dataset

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FlatMap
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import SplitHeartbeats
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, FilterSqi, HasData
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, ComputeSqi, RemoveLowpassTrack, RemoveNan, RemoveSqi, SignalFilter


class BeatSequencePreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int, lowpass_cutoff: int, bandpass_cutoff: Tuple[float, float], min_pressure: int, max_pressure: int, beat_length: int,
                 sequence_steps: int,
                 sequence_stride: int):
        super(BeatSequencePreprocessing, self).__init__([
            HasData(),
            RemoveNan(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            SplitHeartbeats((float32, float32), frequency, beat_length),
            HasData(),
            MakeWindows(sequence_steps, sequence_stride),
            ComputeSqi((float32, float32, float32), 1),
            FilterSqi(0.5, 2),
            RemoveSqi(),
            AddBloodPressureOutput(1),
            RemoveLowpassTrack(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
        ])


class MakeWindows(FlatMap):
    def __init__(self, window_size, step):
        self.window_size = window_size
        self.step = step

    def flatten(self, lowpass_beat: Tensor, bandpass_beat: Tensor) -> Tuple[Dataset, Dataset]:
        return Dataset.from_tensor_slices((lowpass_beat, bandpass_beat)) \
            .window(self.window_size, self.step, drop_remainder=True) \
            .flat_map(lambda low, high: Dataset.zip((low.batch(self.window_size), high.batch(self.window_size))))
