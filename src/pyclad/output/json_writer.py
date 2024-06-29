from typing import List

from pyclad.output.output_writer import OutputWriter, InfoProvider
import itertools


class JsonOutputWriter(OutputWriter):
    def write(self, providers: List[InfoProvider]):
        output_dir = dict(itertools.chain(*(provider.info().items() for provider in providers)))
        print(output_dir)