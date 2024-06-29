import abc
from typing import List, Any, Dict


class InfoProvider(abc.ABC):

    @abc.abstractmethod
    def info(self) -> Dict[str, Any]:
        ...


class OutputWriter(abc.ABC):

    @abc.abstractmethod
    def write(self, providers: List[InfoProvider]):
        ...




