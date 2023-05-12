from pathlib import Path

from mlbpestimation.data.mimic4.mimicdatasetloader import MimicDatasetLoader
from mlbpestimation.data.preprocessedloader import PreprocessedLoader
from mlbpestimation.preprocessing.pipelines.heartbeatpreprocessing import HeartbeatPreprocessing


def main():
    datasets = PreprocessedLoader(MimicDatasetLoader(), HeartbeatPreprocessing(63, beat_length=100)).load_datasets()
    datasets.save(Path('mimic-beat'))


if __name__ == '__main__':
    main()
