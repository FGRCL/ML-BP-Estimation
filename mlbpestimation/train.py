from dotenv import load_dotenv
from hydra import main
from hydra.utils import instantiate
from omegaconf import OmegaConf
from wandb import Settings, init

from mlbpestimation.configuration.train.trainconfiguration import TrainConfiguration

load_dotenv()


@main('configuration/train', 'train', None)
def main(configuration: TrainConfiguration):
    hypothesis = instantiate(configuration.hypothesis)
    init(project=configuration.wandb.project_name,
         entity=configuration.wandb.entity,
         mode=configuration.wandb.mode,
         config=OmegaConf.to_container(configuration, resolve=True),
         settings=Settings(start_method='fork'))  # TODO: check that this is still needed
    hypothesis.train()


if __name__ == '__main__':
    main()
