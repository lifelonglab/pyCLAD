"""Scenarios: the loops that feed a dataset's concepts to a strategy and evaluate it after each one."""

from pyclad.scenarios.concept_agnostic import ConceptAgnosticScenario
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.scenarios.supervised_concept_incremental import (
    SupervisedConceptIncrementalScenario,
)

__all__ = [
    "ConceptAgnosticScenario",
    "ConceptAwareScenario",
    "ConceptIncrementalScenario",
    "SupervisedConceptIncrementalScenario",
]
