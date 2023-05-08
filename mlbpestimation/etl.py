from pathlib import Path

from mlbpestimation.data.preprocessedloader import PreprocessedLoader
from mlbpestimation.data.uci.ucidatasetloader import UciDatasetLoader
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


def main():
    datasets = PreprocessedLoader(UciDatasetLoader(), WindowPreprocessing(125)).load_datasets()
    datasets.save(Path('uci-window'))


if __name__ == '__main__':
    main()
