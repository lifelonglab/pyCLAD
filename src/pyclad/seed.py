"""One entry point for seeding an experiment.

pyCLAD draws randomness from Python's ``random`` module, from NumPy and from PyTorch, depending on the
strategy and the model. :func:`set_seed` seeds all of them, so a script needs a single call for a run to be
repeatable.
"""

import random
from typing import Any, Dict, Optional

import numpy as np

from pyclad.output.output_writer import InfoProvider


class SeedInfo(InfoProvider):
    """The seed an experiment was run with, in a form the output writers accept."""

    def __init__(self, seed: int):
        self.seed = seed

    def info(self) -> Dict[str, Any]:
        return {"seed": self.seed}


def set_seed(seed: int) -> SeedInfo:
    """Seed every source of randomness pyCLAD uses: Python's ``random``, NumPy and, when installed, PyTorch.

    Call it once, before creating the dataset, the model and the strategy. Components that take their own
    ``seed`` or ``rng`` argument keep using it; the ones given none follow this seed.

    This fixes the random draws, not the arithmetic: some GPU operations are not deterministic, so results
    on a GPU can still differ slightly between runs.

    :return: a provider to pass to the output writer, so the seed is saved with the results.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # PyTorch is an optional dependency
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return SeedInfo(seed)


def new_generator(seed: Optional[int] = None) -> np.random.Generator:
    """Create a NumPy generator for a component.

    With a ``seed`` this is ``np.random.default_rng(seed)``. Without one the generator is seeded from
    NumPy's global state, which :func:`set_seed` controls, rather than from fresh entropy as a bare
    ``np.random.default_rng()`` would be. A component that was given no seed is therefore repeatable after
    :func:`set_seed`, and as unpredictable as before without it.
    """
    if seed is None:
        seed = int(np.random.randint(0, np.iinfo(np.int32).max))
    return np.random.default_rng(seed)
