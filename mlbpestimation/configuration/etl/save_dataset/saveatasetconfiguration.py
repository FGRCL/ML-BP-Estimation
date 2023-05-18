from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class SaveDatasetConfiguration:
    dataset: Any
    save_directory: Path


ConfigStore.instance().store(group='save_dataset', name='base_save_dataset_configuration', node=SaveDatasetConfiguration)
