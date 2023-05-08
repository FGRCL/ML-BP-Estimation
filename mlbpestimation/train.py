import argparse
from dataclasses import dataclass

from hydra import main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from mlbpestimation.hypothesis import Hypothesis, hypotheses_repository


@dataclass
class TrainConfig:
    hypothesis: Hypothesis


ConfigStore.instance()


@main(version_base=None, config_path='', config_name='')
def main(configuration: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument('hypothesis', choices=hypotheses_repository.keys(), nargs=1)
    args = parser.parse_args()
    h = hypotheses_repository[args.hypothesis[0]]
    h.train()


if __name__ == '__main__':
    main()
