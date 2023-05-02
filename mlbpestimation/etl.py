from pathlib import Path

from mlbpestimation.data.preprocessedloader import PreprocessedLoader
from mlbpestimation.data.vitaldb.vitaldatasetloader import VitalDatasetLoader
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


def main():
    datasets = PreprocessedLoader(VitalDatasetLoader(), WindowPreprocessing(500)).load_datasets()
    datasets.save(Path(__file__).parent.parent / 'data' / 'vitaldb-window')


if __name__ == '__main__':
    main()
