import tracemalloc
from typing import Any, Dict

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.output.output_writer import InfoProvider


class MemoryUsageCallback(Callback, InfoProvider):
    def __init__(self):
        self._current_memory_usage = {}
        self._peak_memory_usage = {}

    def before_scenario(self, *args, **kwargs):
        tracemalloc.start()

    def after_concept_processing(self, concept: Concept, **kwargs):
        current_size, peak_size = tracemalloc.get_traced_memory()
        self._current_memory_usage[concept.name] = current_size
        self._peak_memory_usage[concept.name] = peak_size
        tracemalloc.reset_peak()

    def after_scenario(self, *args, **kwargs):
        tracemalloc.stop()

    def info(self) -> Dict[str, Any]:
        return {
            "memory_usage_callback": {
                "current_memory_usage": self._current_memory_usage,
                "peak_memory_usage": self._peak_memory_usage,
            }
        }
