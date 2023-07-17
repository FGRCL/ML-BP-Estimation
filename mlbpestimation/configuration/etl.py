from pathlib import Path

from mlbpestimation.configuration.decorators import configuration
from mlbpestimation.configuration.directories.directoriesconfiguration import DirectoriesConfiguration
from mlbpestimation.configuration.hypothesis.hypothesisconfiguration import HypothesisConfiguration


@configuration('base_etl_configuration')
class EtlConfiguration:
    hypothesis: HypothesisConfiguration
    directories: DirectoriesConfiguration
    random_seed: int
    name: Path
