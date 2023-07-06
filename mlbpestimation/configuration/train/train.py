from dataclasses import field
from typing import Dict, List

from omegaconf import MISSING

from mlbpestimation.configuration.decorators import configuration
from mlbpestimation.configuration.train.directories.directoriesconfiguration import DirectoriesConfiguration
from mlbpestimation.configuration.train.hypothesis.hypothesisconfiguration import HypothesisConfiguration
from mlbpestimation.configuration.train.wandb.wandbconfiguration import WandbConfiguration

defaults = [
    {'hypothesis': 'hypothesis'},
    {'wandb': 'disabled'},
    {'directories': 'local'},
]


@configuration('train')
class Train:
    defaults: List[Dict[str, str]] = field(default_factory=lambda: defaults)
    hypothesis: HypothesisConfiguration = MISSING
    wandb: WandbConfiguration = MISSING
    directories: DirectoriesConfiguration = MISSING
    random_seed: int = 106
    job_id: int = '${oc.env:SLURM_JOB_ID,0}'
    array_job_id: int = '${oc.env:SLURM_ARRAY_JOB_ID,0}'
    array_task_id: int = '${oc.env:SLURM_ARRAY_TASK_ID,0}'
    evaluate: bool = True
