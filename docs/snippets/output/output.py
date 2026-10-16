import json
import pathlib

# --8<-- [start:provider]
from typing import Any, Dict

from pyclad.output.json_writer import JsonOutputWriter
from pyclad.output.output_writer import InfoProvider
from pyclad.seed import set_seed


class ExperimentNotes(InfoProvider):
    """Anything else that describes the run, saved next to the results."""

    def __init__(self, description: str, git_commit: str):
        self._description = description
        self._git_commit = git_commit

    def info(self) -> Dict[str, Any]:
        # The top-level key must not collide with another provider's key ("model", "strategy", ...).
        return {"experiment": {"description": self._description, "git_commit": self._git_commit}}


seed = set_seed(7)
notes = ExperimentNotes("baseline run", git_commit="abc1234")
JsonOutputWriter(pathlib.Path("output.json")).write([seed, notes])
# --8<-- [end:provider]

with open("output.json") as f:
    assert json.load(f) == {"seed": 7, "experiment": {"description": "baseline run", "git_commit": "abc1234"}}
