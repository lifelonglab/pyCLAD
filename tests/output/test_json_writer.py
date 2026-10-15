import json
from typing import Any, Dict

import numpy as np

from pyclad.output.json_writer import JsonOutputWriter
from pyclad.output.output_writer import InfoProvider


class _Provider(InfoProvider):
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def info(self) -> Dict[str, Any]:
        return self._payload


def _reject_constant(token: str):
    raise AssertionError(f"{token} is not valid JSON")


def test_undefined_values_are_written_as_null(tmp_path):
    path = tmp_path / "output.json"
    payload = {
        "metrics": {"defined": 0.5, "undefined": float("nan"), "numpy_undefined": np.float32("nan")},
        "matrix": {"a": {"a": 0.9, "b": float("nan")}},
        "infinite": [float("inf"), 1.0],
    }

    JsonOutputWriter(path).write([_Provider(payload)])

    written = json.loads(path.read_text(), parse_constant=_reject_constant)
    assert written["metrics"] == {"defined": 0.5, "undefined": None, "numpy_undefined": None}
    assert written["matrix"] == {"a": {"a": 0.9, "b": None}}
    assert written["infinite"] == [None, 1.0]


def test_numpy_scalars_are_written_as_numbers(tmp_path):
    path = tmp_path / "output.json"

    JsonOutputWriter(path).write([_Provider({"float32": np.float32(0.5), "int64": np.int64(3)})])

    assert json.loads(path.read_text()) == {"float32": 0.5, "int64": 3}
