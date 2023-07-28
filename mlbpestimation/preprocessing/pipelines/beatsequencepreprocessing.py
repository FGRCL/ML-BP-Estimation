from typing import Any, Tuple

from tensorflow import Tensor, float32, reduce_max, reduce_min, stack
from tensorflow.python.ops.array_ops import shape

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation, TransformOperation
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import SplitHeartbeats
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import SlidingWindow
from mlbpestimation.preprocessing.shared.filters import HasData
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import EnsureShape, FilterPressureWithinBounds, FlattenDataset, Reshape, SignalFilter, \
    SqiFiltering, StandardScaling


class BeatSequencePreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int, lowpass_cutoff: int, bandpass_cutoff: Tuple[float, float], min_pressure: int, max_pressure: int, beat_length: int,
                 sequence_steps: int,
                 sequence_stride: int):
        super(BeatSequencePreprocessing, self).__init__([
            FilterHasSignal(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            SplitHeartbeats((float32, float32), frequency, beat_length),
            FilterBeats(sequence_steps),
            SlidingWindow(sequence_steps, sequence_stride),
            SqiFiltering((float32, float32), 0.5, 2),
            HasData(),
            AddBeatSequenceBloodPressure(),
            EnsureShape([None, sequence_steps, beat_length], [None, 2]),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=1),
            Reshape([-1, sequence_steps, beat_length], [-1, 2]),
            FlattenDataset()
        ])


class RemovePressures(TransformOperation):
    def transform(self, beat_sequence: Tensor, pressures: Tensor) -> (Tensor, Tensor):
        return beat_sequence, pressures[:, -1]


class AddBeatSequenceBloodPressure(TransformOperation):
    def transform(self, input_windows: Tensor, output_windows: Tensor) -> Any:
        sbp = reduce_max(output_windows, axis=-1)
        dbp = reduce_min(output_windows, axis=-1)
        sbp, dbp = sbp[:, -1], dbp[:, -1]
        pressures = stack((sbp, dbp), 1)

        return input_windows, pressures


class FilterBeats(FilterOperation):
    def __init__(self, min_beats: int):
        self.min_beats = min_beats

    def filter(self, input_windows: Tensor, output_windows: Tensor) -> bool:
        return shape(input_windows)[0] > self.min_beats
