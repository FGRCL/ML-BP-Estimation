from hydra.utils import instantiate

from mlbpestimation.configuration import configuration


def main():
    hypothesis = instantiate(configuration.hypothesis)
    hypothesis.train()


if __name__ == '__main__':
    main()
