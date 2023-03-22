from pathlib import Path

from tensorflow.python.profiler.profiler_v2 import Profile

from mlbpestimation.data.datasource.mimic4.dataset import MimicDataSource
from mlbpestimation.data.datasource.vitaldb.casesplit import VitalDBDataSource
from mlbpestimation.data.featureset import FeatureSet
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing

data_sources = {
    'mimic': MimicDataSource,
    'vitaldb': VitalDBDataSource,
}

preprocessing_pipelines = {
    'window': WindowPreprocessing,
    'beat': HeartbeatPreprocessing,
}


def main():
    # enable_debug_mode()
    # run_functions_eagerly(True)
    with Profile('logdir'):
        fs = FeatureSet(MimicDataSource(), WindowPreprocessing(63)).build_featuresets(32)
        try:
            fs.save(Path(__file__).parent.parent / 'data' / 'exmaple-test')
        except(RuntimeWarning, UserWarning):
            pass


if __name__ == '__main__':
    main()
