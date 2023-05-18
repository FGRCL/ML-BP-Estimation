from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class WandbConfiguration:
    api_key: str
    project_name: str
    entity: str
    mode: str


ConfigStore.instance().store(group='wandb', name='base_wandb_configuration', node=WandbConfiguration)
