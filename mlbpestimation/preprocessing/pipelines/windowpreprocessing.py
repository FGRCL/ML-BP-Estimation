from tensorflow import bool, float32, reduce_all
from tensorflow.python.data import Options
from tensorflow.python.data.ops.options import AutotuneAlgorithm, AutotuneOptions, ThreadingOptions
from tensorflow.python.ops.array_ops import size

from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, FilterOperation, WithOptions
from mlbpestimation.preprocessing.shared.filters import FilterPressureWithinBounds, FilterSqi
from mlbpestimation.preprocessing.shared.pipelines import FilterHasSignal
from mlbpestimation.preprocessing.shared.transforms import AddBloodPressureOutput, EnsureShape, FlattenDataset, Reshape, SignalFilter, SlidingWindow, StandardScaling


class WindowPreprocessing(DatasetPreprocessingPipeline):
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

    def __init__(self, frequency: int = 500, window_size: int = 8, window_step: int = 2, min_pressure: int = 30,
                 max_pressure: int = 230, lowpass_cutoff=5, bandpass_cutoff=(0.1, 8)):
        window_size_frequency = window_size * frequency
        window_step_frequency = window_step * frequency
        super().__init__([
            FilterHasSignal(),
            FilterSize(window_size_frequency),
            SignalFilter((float32, float32), frequency, lowpass_cutoff, bandpass_cutoff),
            SlidingWindow(window_size_frequency, window_step_frequency),
            FilterSqi((float32, float32), 0.35, 0.8),
            AddBloodPressureOutput(),
            EnsureShape([None, window_size_frequency], [None, 2]),
            FilterPressureWithinBounds(min_pressure, max_pressure),
            StandardScaling(axis=1),
            Reshape([-1, window_size_frequency, 1], [-1, 2]),
            FlattenDataset(),
            WithOptions(self.options)
        ])


class FilterSize(FilterOperation):
    def __init__(self, signal_size: int):
        self.signal_size = signal_size

    def filter(self, *signals) -> bool:
        return reduce_all(size(signals) > self.signal_size)
