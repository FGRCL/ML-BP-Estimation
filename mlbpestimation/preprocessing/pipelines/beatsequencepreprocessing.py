from typing import Tuple

from tensorflow import Tensor, float32

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, TransformOperation
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import SplitHeartbeats
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, HasData
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal, SqiFiltering
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, MakeWindows, RemoveOutputSignal, SetTensorShape, SignalFilter, \
    StandardizeArray


class BeatSequencePreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int, lowpass_cutoff: int, bandpass_cutoff: Tuple[float, float], min_pressure: int, max_pressure: int, beat_length: int,
                 sequence_steps: int,
                 sequence_stride: int):
        super(BeatSequencePreprocessing, self).__init__([
            FilterHasSignal(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            SplitHeartbeats((float32, float32), frequency, beat_length),
            HasData(),
            MakeWindows(sequence_steps, sequence_stride),
            SqiFiltering(0.5, 2, 1),
            AddBloodPressureOutput(1),
            RemoveOutputSignal(),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            RemovePressures(),
            StandardizeArray(1),
            SetTensorShape([sequence_steps, beat_length, 1])
        ])


class RemovePressures(TransformOperation):
    def transform(self, beat_sequence: Tensor, pressures: Tensor) -> (Tensor, Tensor):
        return beat_sequence, pressures[:, -1]
