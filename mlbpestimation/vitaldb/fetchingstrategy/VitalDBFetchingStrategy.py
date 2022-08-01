from abc import ABC, abstractmethod

from numpy import ndarray


class VitalDBFetchingStrategy(ABC):
    @abstractmethod
    def fetch_tracks(self, case_id: int, tracks: list[str], interval: float) -> ndarray:
        pass
