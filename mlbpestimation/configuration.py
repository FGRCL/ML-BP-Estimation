from envyaml import EnvYAML

from mlbpestimation.resources.importer import get_resource

CONFIG_FILE_NAME = 'config.yaml'

config_file = get_resource(CONFIG_FILE_NAME)
with open(config_file, 'r') as config:
    configuration = EnvYAML(config.name)
