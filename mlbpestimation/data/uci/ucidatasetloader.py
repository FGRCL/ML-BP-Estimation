from pathlib import Path

from mat73 import loadmat
from numpy.random import seed, shuffle
from tensorflow.python.data import Dataset
from tensorflow.python.ops.ragged.ragged_factory_ops import constant

from mlbpestimation.configuration import configuration
from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class UciDatasetLoader(DatasetLoader):
    def __init__(self, subsample: float = 1.0):
        self.subsample = subsample
        self.files_directory = Path(configuration['data.directory']) / 'uci'

    def load_datasets(self) -> SplitDataset:
        signals = self._get_abp_list()
        signals = self._shuffle_items(signals)
        signals = self._subsample_items(signals)
        sets = self._make_splits(signals)

        datasets = []
        for set in sets:
            dataset = Dataset.from_tensor_slices(constant(set))
            datasets.append(dataset)

        return SplitDataset(*datasets)

    def _get_abp_list(self):
        signals = []
        for file in self.files_directory.glob('*.mat'):
            mat = loadmat(file)
            for key in mat:
                for record in mat[key]:
                    abp = record[1]
                    signals.append(abp)

        return signals

    # TODO: duplicate code
    def _shuffle_items(self, record_paths):
        seed(configuration['random_seed'])
        shuffle(record_paths)
        return record_paths

    def _subsample_items(self, record_paths):
        nb_records = len(record_paths)
        subsample_size = int(nb_records * self.subsample)
        return record_paths[0:subsample_size]

    def _make_splits(self, record_paths):
        nb_records = len(record_paths)
        return [
            record_paths[:int(nb_records * 0.70)],
            record_paths[int(nb_records * 0.70):int(nb_records * 0.85)],
            record_paths[int(nb_records * 0.85):]
        ]
