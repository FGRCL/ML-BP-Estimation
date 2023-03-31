from abc import ABC, abstractmethod
from typing import List

from numpy import ndarray


class VitalDBFetchingStrategy(ABC):
    @abstractmethod
    def fetch_tracks(self, case_id: int, tracks: List[str], interval: float) -> ndarray:
        pass
