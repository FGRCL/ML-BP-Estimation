from dotenv import load_dotenv
from hydra import main
from hydra.utils import instantiate

from mlbpestimation.configuration.trainconfiguration import TrainConfiguration

load_dotenv()


@main('configuration', 'train_configuration', None)
def main(configuration: TrainConfiguration):
    hypothesis = instantiate(configuration.hypothesis)
    hypothesis.train()


if __name__ == '__main__':
    main()
