from typing import Tuple

from tensorflow import float32 as tfloat32
from tensorflow.python.data.ops.options import AutotuneAlgorithm, AutotuneOptions, Options, ThreadingOptions

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, WithOptions
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, FilterSqi, HasData
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, EnsureShape, FlattenDataset, Reshape, SignalFilter, SplitHeartbeats, StandardScaling


class HeartbeatPreprocessing(DatasetPreprocessingPipeline):
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
                 lowpass_cutoff: float,
                 bandpass_cutoff: Tuple[float, float],
                 min_pressure: int,
                 max_pressure: int,
                 beat_length: int,
                 scale_per_signal: bool,
                 bandpass_input: bool):
        scaling_axis = -1 if scale_per_signal else 1
        dataset_operations = [
            FilterHasSignal(),
            SignalFilter((tfloat32, tfloat32), frequency, lowpass_cutoff, bandpass_cutoff, bandpass_input),
            SplitHeartbeats((tfloat32, tfloat32), frequency, beat_length),
            HasData(),
            FilterSqi((tfloat32, tfloat32), 0.5, 2),
            AddBloodPressureOutput(),
            EnsureShape([None, beat_length], [None, 2]),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=scaling_axis),
            Reshape([-1, beat_length, 1], [-1, 2]),
            FlattenDataset(),
            WithOptions(self.options)
        ]
        super().__init__(dataset_operations)
