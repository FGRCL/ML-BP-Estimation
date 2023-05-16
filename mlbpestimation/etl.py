from hydra import main
from hydra.utils import instantiate

from mlbpestimation.configuration.etl.etlconfiguration import EtlConfiguration
from mlbpestimation.data.datasetloader import DatasetLoader


@main('configuration/etl', 'etl', None)
def main(configuration: EtlConfiguration):
    dataset_loader: DatasetLoader = instantiate(configuration.save_dataset.dataset)
    datasets = dataset_loader.load_datasets()
    datasets.save(configuration.save_dataset.save_directory)


if __name__ == '__main__':
    main()
