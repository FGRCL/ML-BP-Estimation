from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from typing import Any

from mlbpestimation.configuration.train.directories.directoriesconfiguration import DirectoriesConfiguration
from mlbpestimation.configuration.train.wandb.wandbconfiguration import WandbConfiguration


@dataclass
class TrainConfiguration:
    hypothesis: Any
    wandb: WandbConfiguration
    directories: DirectoriesConfiguration
    random_seed: int
    job_id: int
    evaluate: bool


ConfigStore.instance().store(name='base_train_configuration', node=TrainConfiguration)
