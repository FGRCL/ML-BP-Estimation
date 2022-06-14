from vitaldb import VitalFile

from src.vitaldb.fetchingstrategy.VitalDBFetchingStrategy import VitalDBFetchingStrategy


class Sdk(VitalDBFetchingStrategy):
    def fetchvitalfile(self, caseid: int, tracks: list[str]) -> VitalFile:
        return VitalFile(caseid, tracks)
