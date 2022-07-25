from dataclasses import dataclass


@dataclass
class Configuration:
    databaseUrl = "sqlite:///database.db"
