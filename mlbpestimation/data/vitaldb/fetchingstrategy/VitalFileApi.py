from typing import List

from numpy import ndarray
from vitaldb import VitalFile

from mlbpestimation.data.vitaldb.fetchingstrategy.VitalDBFetchingStrategy import VitalDBFetchingStrategy


class VitalFileApi(VitalDBFetchingStrategy):
    def fetch_tracks(self, case_id: int, tracks: List[str], interval: float) -> ndarray:
        return VitalFile(int(case_id), tracks).to_numpy(tracks, interval)
