import abc
from typing import Any, Dict, List


class InfoProvider(abc.ABC):

    @abc.abstractmethod
    def info(self) -> Dict[str, Any]: ...


class OutputWriter(abc.ABC):

    @abc.abstractmethod
    def write(self, providers: List[InfoProvider]): ...
