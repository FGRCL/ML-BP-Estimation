from importlib.resources import path
from pathlib import Path

from typing import ContextManager

resource_package = 'mlbpestimation.resources'


def get_resource(name) -> ContextManager[Path]:
    return path(resource_package, name)
