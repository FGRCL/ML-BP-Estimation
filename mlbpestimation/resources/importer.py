from importlib.resources import path

resource_package = 'mlbpestimation.resources'


def get_resource(name):
    return path(resource_package, name)
