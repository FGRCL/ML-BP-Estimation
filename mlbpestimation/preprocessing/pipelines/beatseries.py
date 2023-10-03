from typing import Tuple

from numpy import float32

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, Prefetch, Shuffle
from mlbpestimation.preprocessing.pipelines.beatsequencepreprocessing import FilterBeats
from mlbpestimation.preprocessing.shared.filters import FilterSqi, HasData
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureSeries, EnsureShape, FilterPressureSeriesWithinBounds, FlattenDataset, Reshape, SignalFilter, SlidingWindow, SplitHeartbeats, StandardScaling, Subsample


class BeatSeriesPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self,
                 frequency: int,
                 lowpass_cutoff: int,
                 bandpass_cutoff: Tuple[float, float],
                 min_pressure: int,
                 max_pressure: int,
                 beat_length: int,
                 sequence_steps: int,
                 sequence_stride: int,
                 scale_per_signal: bool,
                 bandpass_input: bool,
                 random_seed: int,
                 subsample: float):
        scaling_axis = None if scale_per_signal else (1, 2)
        super(BeatSeriesPreprocessing, self).__init__([
            FilterHasSignal(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff, bandpass_input),
            SplitHeartbeats((float32, float32), frequency, beat_length),
            FilterBeats(sequence_steps),
            SlidingWindow(sequence_steps, sequence_stride),
            Subsample((float32, float32), random_seed, subsample),
            HasData(),
            FilterSqi((float32, float32), 0.5, 2),
            HasData(),
            AddBloodPressureSeries(),
            EnsureShape([None, sequence_steps, beat_length], [None, sequence_steps, 2]),
            FilterPressureSeriesWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=scaling_axis),
            Reshape([-1, sequence_steps, beat_length], [-1, sequence_steps, 2]),
            FlattenDataset(),
            Shuffle(),
            Prefetch(),
        ])