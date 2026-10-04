from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Sequence, TypeVar

Item = TypeVar("Item")
Result = TypeVar("Result")

MIN_ITEMS_PER_WORKER = 16


def available_workers() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def map_in_threads(function: Callable[[Item], Result], items: Sequence[Item]) -> List[Result]:
    workers = min(available_workers(), len(items) // MIN_ITEMS_PER_WORKER)
    if workers <= 1:
        return [function(item) for item in items]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, items))
