from dotenv import load_dotenv
from hydra import main
from hydra.utils import instantiate

from mlbpestimation.configuration.train.trainconfiguration import TrainConfiguration

load_dotenv()


@main('configuration/train', 'train', None)
def main(configuration: TrainConfiguration):
    hypothesis = instantiate(configuration.hypothesis)
    hypothesis.train()


if __name__ == '__main__':
    main()
