from mlbpestimation.configuration.decorators import configuration


@configuration('base_wandb_configuration', 'wandb')
class WandbConfiguration:
    api_key: str
    project_name: str
    entity: str
    mode: str
