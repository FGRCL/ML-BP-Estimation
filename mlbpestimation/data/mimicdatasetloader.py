from os.path import splitext
from pathlib import Path
from re import match

import tensorflow
from numpy import empty, split
from numpy.random import choice, seed, shuffle
from tensorflow import TensorSpec, constant, float32, reshape
from tensorflow.python.data import Dataset
from wfdb import rdrecord

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class MimicDatasetLoader(DatasetLoader):
    def __init__(self, mimic_wave_files_directory: str, random_seed: int, subsample: float = 1.0):
        self.mimic_wave_files_directory = Path(mimic_wave_files_directory)
        self.random_seed = random_seed
        self.subsample = subsample

    def load_datasets(self) -> SplitDataset:
        record_paths = self._get_paths()
        record_paths = self._suffle_items(record_paths)
        record_paths = self._subsample_items(record_paths)
        record_paths_splits = self._make_splits(record_paths)

        datasets = []
        for path_split in record_paths_splits:
            dataset = Dataset.from_tensor_slices(path_split) \
                .map(self._tf_read_ap) \
                .map(self._set_shape)
            datasets.append(dataset)

        return SplitDataset(*datasets)

    def _tf_read_ap(self, path):
        return tensorflow.py_function(self._read_abp, [path], TensorSpec([None], float32))

    @staticmethod
    def _read_abp(record_path):
        record = rdrecord(record_path.numpy().decode('ASCII'), channel_names=['ABP'])
        if record.sig_name is not None:
            i = record.sig_name.index('ABP')
            return constant(record.p_signal[:, i], dtype=float32)
        else:
            return empty(0)

    def _set_shape(self, signal):
        return reshape(signal, [-1])

    def _get_paths(self):
        return [splitext(path)[0] for path in
                Path(self.mimic_wave_files_directory).rglob('*hea') if
                match(r'(\d)*.hea', path.name)]

    def _suffle_items(self, record_paths):
        seed(self.random_seed)
        shuffle(record_paths)
        return record_paths

    def _subsample_items(self, record_paths):
        nb_records = len(record_paths)
        subsample_size = int(nb_records * self.subsample)
        return choice(record_paths, subsample_size, False)

    def _make_splits(self, record_paths):
        nb_records = len(record_paths)
        return split(record_paths, [int(nb_records * 0.70), int(nb_records * 0.85)])
