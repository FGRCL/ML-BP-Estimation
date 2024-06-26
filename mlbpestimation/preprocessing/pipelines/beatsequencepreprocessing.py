from typing import Any, Tuple

from tensorflow import Tensor, float32, reduce_max, reduce_min, stack
from tensorflow.python.data.ops.options import AutotuneAlgorithm, AutotuneOptions, Options, ThreadingOptions
from tensorflow.python.ops.array_ops import shape

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation, Prefetch, Shuffle, TransformOperation, WithOptions
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, FilterSqi, HasData
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import EnsureShape, FlattenDataset, Reshape, SignalFilter, SlidingWindow, SplitHeartbeats, StandardScaling


class BeatSequencePreprocessing(DatasetPreprocessingPipeline):
    autotune_options = AutotuneOptions()
    autotune_options.autotune_algorithm = AutotuneAlgorithm.MAX_PARALLELISM
    autotune_options.enabled = True
    autotune_options.ram_budget = int(3.2e10)

    threading_options = ThreadingOptions()
    threading_options.private_threadpool_size = 0

    options = Options()
    options.autotune = autotune_options
    options.deterministic = False
    options.threading = threading_options

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
                 bandpass_input: bool):
        scaling_axis = -1 if scale_per_signal else 2
        super(BeatSequencePreprocessing, self).__init__([
            FilterHasSignal(),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff, bandpass_input),
            SplitHeartbeats((float32, float32), frequency, beat_length),
            FilterBeats(sequence_steps),
            SlidingWindow(sequence_steps, sequence_stride),
            FilterSqi((float32, float32), 0.5, 2),
            HasData(),
            AddBeatSequenceBloodPressure(),
            EnsureShape([None, sequence_steps, beat_length], [None, 2]),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=scaling_axis),
            Reshape([-1, sequence_steps, beat_length], [-1, 2]),
            FlattenDataset(),
            Shuffle(),
            Prefetch(),
            WithOptions(self.options)
        ])


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
