from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

from mlbpestimation.configuration.directories.directoriesconfiguration import DirectoriesConfiguration
from mlbpestimation.configuration.wandb.wandbconfiguration import WandbConfiguration


@dataclass
class TrainConfiguration:
    hypothesis: Any
    wandb: WandbConfiguration
    directories: DirectoriesConfiguration
    random_seed: int


ConfigStore.instance().store(name='base_train_configuration', node=TrainConfiguration)
