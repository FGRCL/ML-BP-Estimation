import logging
import uuid
from shutil import rmtree

from dotenv import load_dotenv
from hydra import main
from hydra.utils import instantiate
from omegaconf import OmegaConf
from wandb import Settings, init

from mlbpestimation.configuration.train import Train
from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.tfrecord.tfrecorddatasetloader import TFRecordDatasetLoader
from mlbpestimation.hypothesis import Hypothesis
from mlbpestimation.models.basemodel import BloodPressureModel

load_dotenv()

log = logging.getLogger(__name__)


@main('configuration', 'train', None)
def main(configuration: Train):
    temp_dataset = uuid.uuid4()
    path = configuration.directories.data / str(temp_dataset)

    try:
        dataset: DatasetLoader = instantiate(configuration.hypothesis.dataset.source)
        if 'decorators' in configuration.hypothesis.dataset:
            for key in configuration.hypothesis.dataset.decorators:
                decorator = configuration.hypothesis.dataset.decorators[key]
                dataset: DatasetLoader = instantiate(decorator, dataset_loader=dataset)

        log.info(f'Cache the dataset for training to {path}')
        datasets = dataset.load_datasets()
        datasets.save(path)
        dataset = TFRecordDatasetLoader(str(configuration.directories.data), str(temp_dataset))
        log.info('Done caching the dataset')

        model: BloodPressureModel = instantiate(configuration.hypothesis.model)
        optimization = instantiate(configuration.hypothesis.optimization)
        hypothesis = Hypothesis(dataset, model, configuration.hypothesis.output_directory, optimization)

        init(project=configuration.wandb.project_name,
             entity=configuration.wandb.entity,
             mode=configuration.wandb.mode,
             config=OmegaConf.to_container(configuration, resolve=True),
             settings=Settings(start_method="thread"),
             )
        hypothesis.train()
        if configuration.evaluate:
            hypothesis.evaluate()

    finally:
        log.info(f'Cleaning up the dataset {path}')
        rmtree(path)


if __name__ == '__main__':
    main()
