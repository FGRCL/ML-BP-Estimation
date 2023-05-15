from hydra.utils import instantiate

from mlbpestimation.conf import configuration


def main():
    hypothesis = instantiate(configuration.hypothesis)
    hypothesis.train()


if __name__ == '__main__':
    main()
