from os.path import splitext
from pathlib import Path
from random import Random
from re import match

from numpy import split
from tensorflow import TensorSpec, float32
from tensorflow.python.data import Dataset

from mlbpestimation.configuration import configuration
from mlbpestimation.data.datasource.database import Database
from mlbpestimation.data.datasource.mimic4.generator import MimicCaseGenerator
from mlbpestimation.data.multipartdataset import MultipartDataset

SEED = 106


class MimicDataSource(Database):
    def get_datasets(self) -> MultipartDataset:
        record_paths = get_paths()
        Random(SEED).shuffle(record_paths)
        nb_records = len(record_paths)
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
