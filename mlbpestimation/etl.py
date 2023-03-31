from pathlib import Path

from tensorflow.python.profiler.profiler_v2 import Profile

from mlbpestimation.data.datasource.mimic4.mimicdatabase import MimicDatabase
from mlbpestimation.data.datasource.vitaldb.vitaldatabase import VitalDatabase
from mlbpestimation.data.featureset import FeatureSet
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing

data_sources = {
    'mimic': MimicDatabase,
    'vitaldb': VitalDatabase,
}

preprocessing_pipelines = {
    'window': WindowPreprocessing,
    'beat': HeartbeatPreprocessing,
}


def main():
    with Profile('logdir'):
        fs = FeatureSet(MimicDatabase(), WindowPreprocessing(63)).build_featuresets()
        try:
            fs.save(Path(__file__).parent.parent / 'data' / 'example-test')
        except(RuntimeWarning, UserWarning):
            pass


if __name__ == '__main__':
    main()
