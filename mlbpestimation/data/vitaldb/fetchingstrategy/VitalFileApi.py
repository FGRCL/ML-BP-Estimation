from numpy import ndarray
from vitaldb import VitalFile

from mlbpestimation.data.vitaldb.fetchingstrategy.VitalDBFetchingStrategy import VitalDBFetchingStrategy


class VitalFileApi(VitalDBFetchingStrategy):
    def fetch_tracks(self, case_id: int, tracks: list[str], interval: float) -> ndarray:
        return VitalFile(case_id, tracks).to_numpy(tracks, interval)
