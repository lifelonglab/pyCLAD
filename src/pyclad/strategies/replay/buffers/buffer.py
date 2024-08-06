import abc
from typing import Any, Dict

import numpy as np

from pyclad.output.output_writer import InfoProvider


class ReplayBuffer(InfoProvider, abc.ABC):

    @abc.abstractmethod
    def update(self, data: np.ndarray) -> None: ...

    @abc.abstractmethod
    def data(self) -> np.ndarray: ...

    @abc.abstractmethod
    def name(self) -> str: ...

    def info(self) -> Dict[str, Any]:
        return {"name": self.name(), **self.additional_info(), "buffer_size": len(self.data())}

    def additional_info(self) -> Dict:
        return {}
