import tensorflow
from tensorflow import float64

from src.preprocessing.filters import has_data, pressure_within_bounds
from src.preprocessing.pipelines.base import DatasetPreprocessingPipeline, TransformOperation, FilterOperation, NumpyTransformOperation
from src.preprocessing.transforms import remove_nan, abp_low_pass, extract_clean_windows, to_tensor, \
    extract_sbp_dbp_from_abp_window, scale_array, print_tensor


class WindowPreprocessing(DatasetPreprocessingPipeline):
    def __init__(self, frequency: int, window_size: int, window_step: int, min_pressure: int, max_pressure: int):
        dataset_operations = [
            FilterOperation(has_data),
            TransformOperation(remove_nan),
            NumpyTransformOperation(abp_low_pass, float64, frequency),
            NumpyTransformOperation(extract_clean_windows, float64, frequency, window_size, window_step),
            FilterOperation(pressure_within_bounds, min_pressure, max_pressure),
            TransformOperation(extract_sbp_dbp_from_abp_window),
            TransformOperation(scale_array)
        ]
        super().__init__(dataset_operations)
