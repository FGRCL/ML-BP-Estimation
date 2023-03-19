import argparse

from mlbpestimation.data.datasource.mimic4.dataset import MimicDataSource
from mlbpestimation.data.featureset import FeatureSet
from mlbpestimation.hypothesis import hypotheses_repository
from mlbpestimation.preprocessing.pipelines.windowpreprocessing import WindowPreprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hypothesis', choices=hypotheses_repository.keys(), nargs=1)
    args = parser.parse_args()

    for test in FeatureSet(MimicDataSource(), WindowPreprocessing()).build_featuresets(32).train:
        print(test)
    h = hypotheses_repository[args.hypothesis[0]]
    h.train()


if __name__ == '__main__':
    main()
