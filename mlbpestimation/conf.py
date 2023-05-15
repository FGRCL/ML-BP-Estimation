from dotenv import load_dotenv
from hydra import compose, initialize

from mlbpestimation.configuration.trainconfiguration import TrainConfiguration

load_dotenv()

initialize(version_base=None, config_path='configuration')
configuration: TrainConfiguration = compose(config_name='train_configuration')
