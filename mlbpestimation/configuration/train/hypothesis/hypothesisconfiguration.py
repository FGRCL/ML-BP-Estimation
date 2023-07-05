from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING

from mlbpestimation.configuration.train.hypothesis.dataset.datasetconfiguration import DatasetConfiguration


@dataclass
class HypothesisConfiguration:
    _target_: str = 'mlbpestimation.hypothesis.Hypothesis'
    dataset: DatasetConfiguration = MISSING
    model: DictConfig = MISSING
    optimization: DictConfig = MISSING
    output_directory: str = '${directories.output}'


ConfigStore.instance().store(group='hypothesis', name='base_hypothesis_configuration', node=HypothesisConfiguration)
