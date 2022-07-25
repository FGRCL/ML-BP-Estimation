from os.path import splitext
from pathlib import Path
from random import Random
from re import match

from numpy import split
from tensorflow import TensorSpec, float32
from tensorflow.python.data import Dataset

from src.data.mimic4.generator import MimicCaseGenerator

SEED = 106


def load_mimic_dataset() -> (Dataset, Dataset, Dataset):
    record_paths = get_paths()
    Random(SEED).shuffle(record_paths)
    nb_records = len(record_paths)
    record_paths_splits = split(record_paths, [int(nb_records * 0.70), int(nb_records * 0.85)])
    datasets = []
    for path_split in record_paths_splits:
        datasets.append(
            Dataset.from_generator(
                lambda: MimicCaseGenerator(path_split),
                output_signature=(
                    TensorSpec(shape=(None, 1), dtype=float32)
                )
            )
        )

    return datasets


def get_paths():
    return [splitext(path)[0] for path in
            Path('../../../data/mimic-IV/physionet.org/files/mimic4wdb/0.1.0/waves').rglob('*hea') if
            match(r'(\d)*.hea', path.name)]
