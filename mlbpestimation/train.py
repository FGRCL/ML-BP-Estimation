from dotenv import load_dotenv
from hydra import main
from hydra.utils import instantiate
from omegaconf import OmegaConf
from wandb import Settings, init

from mlbpestimation.configuration.train import Train
from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.hypothesis import Hypothesis
from mlbpestimation.models.basemodel import BloodPressureModel

load_dotenv()


@main('configuration', 'train', None)
def main(configuration: Train):
    dataset: DatasetLoader = instantiate(configuration.hypothesis.dataset.source)
    if 'decorators' in configuration.hypothesis.dataset:
        for key in configuration.hypothesis.dataset.decorators:
            decorator = configuration.hypothesis.dataset.decorators[key]
            dataset: DatasetLoader = instantiate(decorator, dataset_loader=dataset)
    model: BloodPressureModel = instantiate(configuration.hypothesis.model)
    optimization = instantiate(configuration.hypothesis.optimization)
    hypothesis = Hypothesis(dataset, model, configuration.hypothesis.output_directory, optimization)

    init(project=configuration.wandb.project_name,
         entity=configuration.wandb.entity,
         mode=configuration.wandb.mode,
         config=OmegaConf.to_container(configuration, resolve=True),
         settings=Settings(start_method='fork'))  # TODO: check that this is still needed
    hypothesis.train()
    if configuration.evaluate:
        hypothesis.evaluate()


if __name__ == '__main__':
    main()
