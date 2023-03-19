from typing import List

import vitaldb
from numpy import empty, ndarray

from mlbpestimation.data.datasource.vitaldb.fetchingstrategy.VitalDBFetchingStrategy import VitalDBFetchingStrategy


class DatasetApi(VitalDBFetchingStrategy):
    def fetch_tracks(self, case_id: int, tracks: List[str], interval: float) -> ndarray:
        tracks = vitaldb.load_case(case_id, tracks, interval)

        if tracks.size == 0:
            tracks = empty([0, 1])

        return tracks
