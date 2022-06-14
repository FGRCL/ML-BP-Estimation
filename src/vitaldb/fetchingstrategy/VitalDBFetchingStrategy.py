from abc import ABC, abstractmethod

from vitaldb import VitalFile


class VitalDBFetchingStrategy(ABC):
    @abstractmethod
    def fetchvitalfile(self, caseid: int, tracks: list[str]) -> VitalFile:
        pass
