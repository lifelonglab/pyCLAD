import itertools
import json
import pathlib
from typing import List

from pyclad.output.output_writer import InfoProvider, OutputWriter


class JsonOutputWriter(OutputWriter):
    def __init__(self, path: pathlib.Path):
        self._path = path

    def write(self, providers: List[InfoProvider]):
        output_data = dict(itertools.chain(*(provider.info().items() for provider in providers)))
        with open(self._path, "w") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4, default=lambda o: "<not serializable>")
