from dataclasses import dataclass

from vitaldb import VitalFile

from src.vital.fetchingstrategy.VitalDBFetchingStrategy import VitalDBFetchingStrategy

@dataclass
class VitalFileOptions:
    tracks: list[str]
    interval: float
    return_datetime: bool = False
    return_timestamp: bool = True


class VitalDBGenerator(object):

    def __init__(self, options: VitalFileOptions, fetchingstrategy: VitalDBFetchingStrategy, maxcase=6388):
        self.caseId = 1
        self.maxcase = maxcase
        self.options = options
        self.fetchingstrategy = fetchingstrategy

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self) -> VitalFile:
        if self.caseId <= self.maxcase:
            file = self.fetchingstrategy.fetchvitalfile(self.caseId, self.options.tracks).to_numpy(self.options.tracks, self.options.interval, self.options.return_datetime, self.options.return_timestamp)
            self.caseId = self.caseId + 1
            return file
        raise StopIteration()
