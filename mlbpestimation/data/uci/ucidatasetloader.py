from pathlib import Path

from mat73 import loadmat
from numpy.random import seed, shuffle
from tensorflow.python.data import Dataset
from tensorflow.python.ops.ragged.ragged_concat_ops import stack

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset


class UciDatasetLoader(DatasetLoader):
    def __init__(self, uci_files_directory: str, frequency: int, random_seed: int, subsample: float = 1.0, use_ppg: bool = False):
        self.subsample = subsample
        self.frequency = frequency
        self.random_seed = random_seed
        self.uci_files_directory = Path(uci_files_directory)
        self.input_index = 0 if use_ppg else 1

    def load_datasets(self) -> SplitDataset:
        signals = self._get_abp_list()
        signals = self._shuffle_items(signals)
        signals = self._subsample_items(signals)
        splits = self._make_splits(signals)

        datasets = []
        for split in splits:
            input_signals = stack([s[0] for s in split])
            output_signals = stack([s[1] for s in split])
            dataset = Dataset.from_tensor_slices((input_signals, output_signals))
            datasets.append(dataset)

        return SplitDataset(*datasets)

    def _get_abp_list(self):
        signals = []
        for file in self.uci_files_directory.glob('*.mat'):
            mat = loadmat(file)
            for key in mat:
                for record in mat[key]:
                    input_signal = record[self.input_index]
                    abp = record[1]
                    signals.append([input_signal, abp])

        return signals

    # TODO: duplicate code
    def _shuffle_items(self, record_paths):
        seed(self.random_seed)
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
