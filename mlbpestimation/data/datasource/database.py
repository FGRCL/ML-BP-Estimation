from abc import ABC, abstractmethod


class Database(ABC):

    @abstractmethod
    def get_datasets(self):
        ...
