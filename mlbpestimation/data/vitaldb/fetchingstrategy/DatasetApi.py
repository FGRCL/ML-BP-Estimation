from typing import List

import vitaldb
from numpy import empty, ndarray

from mlbpestimation.data.vitaldb.fetchingstrategy.VitalDBFetchingStrategy import VitalDBFetchingStrategy


class DatasetApi(VitalDBFetchingStrategy):
    def fetch_tracks(self, case_id: int, track_names: List[str], interval: float) -> ndarray:
        tracks = vitaldb.load_case(case_id, track_names, interval)

        if tracks.size < len(track_names):
            tracks = empty((1, len(track_names)))

        return tracks
