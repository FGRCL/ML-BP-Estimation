from mlbpestimation.configuration.decorators import configuration
from mlbpestimation.configuration.train.directories.directoriesconfiguration import DirectoriesConfiguration
from mlbpestimation.configuration.train.hypothesis.hypothesisconfiguration import HypothesisConfiguration
from mlbpestimation.configuration.train.wandb.wandbconfiguration import WandbConfiguration


@configuration('base_train_configuration')
class Train:
    hypothesis: HypothesisConfiguration
    wandb: WandbConfiguration
    directories: DirectoriesConfiguration
    random_seed: int
    job_id: int
    array_job_id: int
    array_task_id: int
    evaluate: bool
