from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class DatasetConfiguration:
    source: DictConfig
    decorators: DictConfig


ConfigStore.instance().store(group='hypothesis/dataset', name='base_dataset_configuration', node=DatasetConfiguration)
