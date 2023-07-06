from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class DatasetConfiguration:
    source: Any
    decorators: Any


ConfigStore.instance().store(group='hypothesis/dataset', name='base_dataset_configuration', node=DatasetConfiguration)
