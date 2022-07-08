import tensorflow as tf
from tensorflow.python.data import Dataset

from src.preprocessing.pipelines.datasetpreprocessor import DatasetPreprocessor
from src.preprocessing.transforms import extract_abp_track, remove_nan, split_heartbeats, \
    extract_sbp_dbp_from_heartbeat, remove_lowpass_track, standardize_track, filter_track
from src.preprocessing.filters import has_data


class HeartbeatPreprocessor(DatasetPreprocessor):
    def __init__(self, frequency: int):
        self.frequency = frequency

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        dataset = dataset.filter(has_data)
        dataset = dataset.map(extract_abp_track)
        dataset = dataset.map(remove_nan)
        dataset = dataset.map(lambda x: filter_track(x, self.frequency))
        dataset = dataset.map(split_heartbeats)
        dataset = dataset.map(extract_sbp_dbp_from_heartbeat)
        dataset = dataset.map(remove_lowpass_track)
        dataset = dataset.map(standardize_track)

        return dataset

    def _filter_track_adapter(self):
        return lambda x: filter_track(x, self.frequency)
