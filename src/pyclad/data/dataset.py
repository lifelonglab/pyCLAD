from abc import abstractmethod
from typing import Any, Dict

from pyclad.output.output_writer import InfoProvider


class Dataset(InfoProvider):

    @abstractmethod
    def name(self) -> str: ...

    def additional_info(self) -> Dict[str, Any]:
        return {}

    def info(self) -> Dict[str, Any]:
        return {"dataset": {"name": self.name(), **self.additional_info()}}
