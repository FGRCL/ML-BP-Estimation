from importlib.resources import files

resource_package = 'mlbpestimation.resources'


def get_resource(name):
    return files(resource_package).joinpath(name).
