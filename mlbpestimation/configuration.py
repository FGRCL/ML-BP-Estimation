from envyaml import EnvYAML

from mlbpestimation.resources.importer import get_resource

CONFIG_FILE_NAME = 'config.yaml'

with get_resource(CONFIG_FILE_NAME) as config_file_path:
    configuration = EnvYAML(str(config_file_path))
