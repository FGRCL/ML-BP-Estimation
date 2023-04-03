from numpy import arange, split
from tensorflow import TensorSpec, float32
from tensorflow.python.data import Dataset

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset
from mlbpestimation.data.vitaldb.casegenerator import MAX_VITAL_DB_CASE, MIN_VITAL_DB_CASE, VitalDBGenerator, \
    VitalFileOptions
from mlbpestimation.data.vitaldb.fetchingstrategy.DatasetApi import DatasetApi


class VitalDatasetLoader(DatasetLoader):

    def load_datasets(self) -> SplitDataset:
        options = VitalFileOptions(
            ['SNUADC/ART'],
            1 / 500
        )

        case_ids = arange(MIN_VITAL_DB_CASE, MAX_VITAL_DB_CASE + 1)
        nb_cases = len(case_ids)
        case_id_splits = split(case_ids, [int(nb_cases * 0.70), int(nb_cases * 0.85)])

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
