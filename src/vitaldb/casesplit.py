import numpy as np
from zlib import crc32
from src.vitaldb.casegenerator import VitalDBGenerator, VitalFileOptions, VitalDBFetchingStrategy, MIN_VITAL_DB_CASE, \
    MAX_VITAL_DB_CASE


def get_splits(split_percentages: list[float], case_range: list[int] = range(MIN_VITAL_DB_CASE, MAX_VITAL_DB_CASE + 1)) -> list[list[int]]:
    if not round(sum(split_percentages), 2) == 1:
        raise Exception(f'split percentages should sum up to 100%, but summed up to {sum(split_percentages) * 100}%')

    split_thresholds = [0]
    cumulative_percentage = split_thresholds[0]
    for split_percentage in split_percentages:
        cumulative_percentage = cumulative_percentage + split_percentage
        split_thresholds.append(cumulative_percentage * 2 ** 32)

    splits = [[] for x in split_percentages]
    for case in case_range:
        hashed_id = crc32(np.int64(case)) & 0xffffffff  # TODO this hash doesn't provide the most reliable dataset sizes. consider reworking it
        for i in range(0, len(split_thresholds) - 1):
            if split_thresholds[i] < hashed_id <= split_thresholds[i + 1]:
                splits[i].append(case)
    return splits
