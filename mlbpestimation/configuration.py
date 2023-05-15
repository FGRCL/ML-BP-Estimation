from dotenv import load_dotenv
from hydra import compose, initialize

load_dotenv()

initialize(version_base=None, config_path='conf')
configuration = compose(config_name='train_configuration')
