from dataclasses import dataclass

from numpy import ndarray

from src.vitaldb.fetchingstrategy.VitalDBFetchingStrategy import VitalDBFetchingStrategy


@dataclass
class VitalFileOptions:
    tracks: list[str]
    interval: float


class VitalDBGenerator(object):

    def __init__(self, options: VitalFileOptions, fetching_strategy: VitalDBFetchingStrategy, case_ids: list[int]):
        if len(case_ids) < 1 or len(case_ids) > 6388:  # TODO save those constants somewhere
            raise Exception(f'The generator should have between 1 and 6388 case ids but got {len(case_ids)}')
        self.case_ids = iter(case_ids)
        self.options = options
        self.fetching_strategy = fetching_strategy

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self) -> ndarray:
        return self.fetching_strategy.fetch_tracks(next(self.case_ids), self.options.tracks, self.options.interval)
