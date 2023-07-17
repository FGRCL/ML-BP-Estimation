from dataclasses import dataclass
from pathlib import Path

from hydra.core.config_store import ConfigStore


@dataclass
class DirectoriesConfiguration:
    data: Path
    output: Path


ConfigStore.instance().store(group='directories', name='base_directories_configuration', node=DirectoriesConfiguration)
