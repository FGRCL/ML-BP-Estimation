from pathlib import Path

from mlbpestimation.data.mimic4.mimicdatabase import MimicDatasetLoader
from mlbpestimation.data.preprocessedloader import PreprocessedLoader
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


def main():
    datasets = PreprocessedLoader(MimicDatasetLoader(), WindowPreprocessing(63)).load_datasets()
    datasets.save(Path(__file__).parent.parent / 'data' / 'mimic-window')


if __name__ == '__main__':
    main()
