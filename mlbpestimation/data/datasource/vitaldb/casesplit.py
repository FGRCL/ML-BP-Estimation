from typing import List
from zlib import crc32

import numpy as np
from tensorflow import TensorSpec, float32
from tensorflow.python.data import Dataset

from mlbpestimation.data.datasource.database import Database
from mlbpestimation.data.datasource.vitaldb.casegenerator import MAX_VITAL_DB_CASE, MIN_VITAL_DB_CASE, VitalDBGenerator, \
    VitalFileOptions
from mlbpestimation.data.datasource.vitaldb.fetchingstrategy.DatasetApi import DatasetApi
from mlbpestimation.data.multipartdataset import MultipartDataset


class VitalDBDataSource(Database):

    def get_datasets(self) -> MultipartDataset:
        options = VitalFileOptions(
            ['SNUADC/ART'],
            1 / 500
        )

        case_splits = get_splits([0.7, 0.15, 0.15])

        datasets = []
        for case_split in case_splits:
            datasets.append(
                Dataset.from_generator(
                    lambda c=case_split: VitalDBGenerator(options, DatasetApi(), c),
                    output_signature=(
                        TensorSpec(shape=(None, 1), dtype=float32)
                    )
                )
            )

        return MultipartDataset(*datasets)


def get_splits(split_percentages: List[float],
               case_range: List[int] = range(MIN_VITAL_DB_CASE, MAX_VITAL_DB_CASE + 1)) -> List[List[int]]:
    if not round(sum(split_percentages), 2) == 1:
        raise Exception(f'split percentages should sum up to 100%, but summed up to {sum(split_percentages) * 100}%')

    split_thresholds = [0]
    cumulative_percentage = split_thresholds[0]
    for split_percentage in split_percentages:
        cumulative_percentage = cumulative_percentage + split_percentage
        split_thresholds.append(cumulative_percentage * 2 ** 32)

    splits = [[] for x in split_percentages]
    for case in case_range:
        hashed_id = crc32(np.int64(
            case)) & 0xffffffff  # TODO this hash doesn't provide the most reliable dataset sizes. consider reworking it
        for i in range(0, len(split_thresholds) - 1):
            if split_thresholds[i] < hashed_id <= split_thresholds[i + 1]:
                splits[i].append(case)
    return splits
