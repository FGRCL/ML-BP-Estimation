from dataclasses import dataclass
from pathlib import Path

from hydra.core.config_store import ConfigStore

from mlbpestimation.configuration.etl.save_dataset.saveatasetconfiguration import SaveDatasetConfiguration


@dataclass
class EtlConfiguration:
    save_dataset: SaveDatasetConfiguration
    data_directory: Path
    random_seed: int


ConfigStore.instance().store(name='base_etl_configuration', node=EtlConfiguration)
