from numpy import arange, split
from numpy.random import choice, seed, shuffle
from tensorflow import TensorSpec, float32
from tensorflow.python.data import Dataset

from mlbpestimation.configuration import configuration
from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset
from mlbpestimation.data.vitaldb.casegenerator import MAX_VITAL_DB_CASE, MIN_VITAL_DB_CASE, VitalDBGenerator, \
    VitalFileOptions
from mlbpestimation.data.vitaldb.fetchingstrategy.DatasetApi import DatasetApi


class VitalDatasetLoader(DatasetLoader):
    def __init__(self, subsample: float = 1.0):
        self.subsample = subsample

    def load_datasets(self) -> SplitDataset:
        options = VitalFileOptions(
            ['SNUADC/ART'],
            1 / 500
        )

        case_ids = self._get_random_case_id_list()
        case_ids = self._subsample_items(case_ids)
        case_id_splits = self._make_splits(case_ids)

        datasets = []
        for case_split in case_id_splits:
            datasets.append(
                Dataset.from_generator(
                    lambda c=case_split: VitalDBGenerator(options, DatasetApi(), c),
                    output_signature=(
                        TensorSpec(shape=(None, 1), dtype=float32)
                    )
                )
            )

        return SplitDataset(*datasets)

    def _get_random_case_id_list(self):
        case_ids = arange(MIN_VITAL_DB_CASE, MAX_VITAL_DB_CASE + 1)
        seed(configuration['random_seed'])
        shuffle(case_ids)
        return case_ids

    def _subsample_items(self, case_ids):
        nb_cases = len(case_ids)
        subsample_size = int(nb_cases * self.subsample)
        return choice(case_ids, subsample_size, False)

    def _make_splits(self, case_ids):
        nb_cases = len(case_ids)
        return split(case_ids, [int(nb_cases * 0.70), int(nb_cases * 0.85)])
