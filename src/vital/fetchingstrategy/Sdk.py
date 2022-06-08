from vitaldb import VitalFile

from src.vital.fetchingstrategy.VitalDBFetchingStrategy import VitalDBFetchingStrategy


class Sdk(VitalDBFetchingStrategy):
    def fetchvitalfile(self, caseid: int, tracks: list[str]) -> VitalFile:
        return VitalFile(caseid, tracks)
