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

    def __init__(self, frequency=500, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8), min_pressure=30, max_pressure=230,
                 beat_length=400):
        dataset_operations = [
            FilterHasSignal(),
            SignalFilter((tfloat32, tfloat32), frequency, lowpass_cutoff, bandpass_cutoff),
            SplitHeartbeats((tfloat32, tfloat32), frequency, beat_length),
            HasData(),
            FilterSqi((tfloat32, tfloat32), 0.5, 2),
            AddBloodPressureOutput(),
            EnsureShape([None, beat_length], [None, 2]),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=1),
            Reshape([-1, beat_length, 1], [-1, 2]),
            FlattenDataset(),
            WithOptions(self.options)
        ]
        super().__init__(dataset_operations)
