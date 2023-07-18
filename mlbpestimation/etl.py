from dotenv import load_dotenv
from hydra import main
from hydra.utils import instantiate

from mlbpestimation.configuration.etl import EtlConfiguration
from mlbpestimation.data.datasetloader import DatasetLoader

load_dotenv()


@main('configuration', 'etl', None)
def main(configuration: EtlConfiguration):
    dataset: DatasetLoader = instantiate(configuration.hypothesis.dataset.source)
    if 'decorators' in configuration.hypothesis.dataset:
        for key in configuration.hypothesis.dataset.decorators:
            decorator = configuration.hypothesis.dataset.decorators[key]
            dataset: DatasetLoader = instantiate(decorator, dataset_loader=dataset)

    datasets = dataset.load_datasets()
    datasets.save(configuration.directories.data / configuration.name)


if __name__ == '__main__':
    main()
