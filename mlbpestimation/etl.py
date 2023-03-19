from pathlib import Path

from mlbpestimation.data.datasource.vitaldb.casesplit import VitalDBDataSource
from mlbpestimation.data.featureset import FeatureSet
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


def main():
    # fs = FeatureSet(VitalDBDataSource(), HeartbeatPreprocessing()).build_featuresets(20)
    fs = FeatureSet(VitalDBDataSource(), WindowPreprocessing()).build_featuresets(32)
    fs.save(Path(__file__).parent.parent / 'data' / 'exmaple-test')


if __name__ == '__main__':
    main()
