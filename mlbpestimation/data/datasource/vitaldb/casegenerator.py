from dataclasses import dataclass
from typing import Iterable, List

from numpy import ndarray

from mlbpestimation.data.datasource.vitaldb.fetchingstrategy.VitalDBFetchingStrategy import VitalDBFetchingStrategy

MIN_VITAL_DB_CASE = 1
MAX_VITAL_DB_CASE = 6388


@dataclass
class VitalFileOptions:
    tracks: List[str]
    interval: float


class VitalDBGenerator(object):

    def __init__(self, options: VitalFileOptions, fetching_strategy: VitalDBFetchingStrategy, case_ids: Iterable[int]):
        if len(case_ids) < 1:
            raise Exception(f'No case ids were passed to the generator')
        if min(case_ids) < MIN_VITAL_DB_CASE or max(case_ids) > MAX_VITAL_DB_CASE:
            raise Exception(f'At least one case was outside the allowed range for case ids')
        self.case_ids = iter(case_ids)
        self.options = options
        self.fetching_strategy = fetching_strategy

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self) -> ndarray:
        return self.fetching_strategy.fetch_tracks(next(self.case_ids), self.options.tracks, self.options.interval)
