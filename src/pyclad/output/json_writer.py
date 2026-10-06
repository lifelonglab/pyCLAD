import itertools
import json
import math
import pathlib
from typing import Any, List

import numpy as np

from pyclad.output.output_writer import InfoProvider, OutputWriter


class JsonOutputWriter(OutputWriter):
    def __init__(self, path: pathlib.Path):
        self._path = path

    def write(self, providers: List[InfoProvider]):
        output_data = dict(itertools.chain(*(provider.info().items() for provider in providers)))
        with open(self._path, "w") as f:
            json.dump(
                _json_ready(output_data),
                f,
                ensure_ascii=False,
                indent=4,
                allow_nan=False,
                default=lambda o: "<not serializable>",
            )


def _json_ready(value: Any) -> Any:
    """Replace values the JSON format cannot hold.

    An undefined metric is NaN, which ``json`` would write as a bare ``NaN`` token that other JSON
    readers reject; it is written as ``null`` instead, and so are infinities. NumPy scalars become
    the Python numbers they wrap.
    """
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
