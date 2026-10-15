from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Concept:
    """A named part of a dataset: one distribution, activity or task.

    :param name: identifies the concept; train and test concepts with the same name belong together.
    :param data: the samples, with the first axis running over samples.
    :param labels: one label per sample, ``0`` for a normal sample and ``1`` for an anomaly. Optional for
        train concepts, which unsupervised strategies treat as normal data; required for test concepts.
    """

    name: str
    data: np.array
    labels: Optional[np.array] = None
