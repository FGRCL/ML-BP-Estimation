from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


def configuration(name, group=None):
    def decorator(config_class: type):
        config_class = dataclass(config_class)
        cs.store(name, config_class, group)

        return config_class

    return decorator
