from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig


@main(version_base=None, config_path='conf', config_name='train_configuration')
def main(configuration: DictConfig):
    hypothesis = instantiate(configuration.hypothesis)
    hypothesis.train()


if __name__ == '__main__':
    main()
