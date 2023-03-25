from os.path import splitext
from pathlib import Path
from random import Random
from re import match

from numpy import split
from numpy.random import choice
from tensorflow import TensorSpec, float32
from tensorflow.python.data import Dataset

from mlbpestimation.configuration import configuration
from mlbpestimation.data.datasource.database import Database
from mlbpestimation.data.datasource.mimic4.generator import MimicCaseGenerator
from mlbpestimation.data.multipartdataset import MultipartDataset

SEED = 106


class MimicDatabase(Database):
    def __init__(self, subsample: float = 1.0):
        self.subsample = subsample

    def get_datasets(self) -> MultipartDataset:
        record_paths = get_paths()
        Random(SEED).shuffle(record_paths)
        nb_records = len(record_paths)
        subsample_size = int(nb_records * self.subsample)
        record_paths = choice(record_paths, subsample_size, False)
        record_paths_splits = split(record_paths, [int(nb_records * 0.70), int(nb_records * 0.85)])

        datasets = []
        for path_split in record_paths_splits:
            datasets.append(
                Dataset.from_generator(
                    lambda p=path_split: MimicCaseGenerator(p),
                    output_signature=(
                        TensorSpec(shape=(None,), dtype=float32)
                    )
                )
            )

        return MultipartDataset(*datasets)


def get_paths():
    return [splitext(path)[0] for path in
            Path(configuration['data.mimic.file_location']).rglob('*hea') if
            match(r'(\d)*.hea', path.name)]
