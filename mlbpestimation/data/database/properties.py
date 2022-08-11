from mlbpestimation.configuration import configuration


def build_url():
    protocol = configuration['data.database.url.protocol']
    username = configuration['data.database.url.username']
    password = configuration['data.database.url.password']
    host = configuration['data.database.url.host']
    port = configuration['data.database.url.port']
    schema = configuration['data.database.url.schema']
    return f'{protocol}://{username}:{password}@{host}:{port}/{schema}'


database_url = build_url()
