"""Step-schedule grouping of concepts into multi-category training steps.

Instead of the default "one category = one training step", consecutive concepts are
merged into a single training step according to a schedule such as ``"14-1"``,
``"10-5"``, ``"3x5"`` or ``"10-1x5"``. Training then walks over ``T`` steps while
evaluation stays per category (``N``), which is how CDAD-style continual anomaly
detection papers report results.

The side effect is that the concept-level metric matrix stops being square: it becomes
``T x N`` with ``T < N``. Metrics that assume a square matrix (``ContinualAverage``,
``BackwardTransfer``, ``ForwardTransfer``, ``ForgettingMeasure``) reject such a matrix;
use the schedule-aware metrics in :mod:`pyclad.metrics.continual` instead.

Order of operations matters: **ordering -> grouping -> first_seen_step**. Concepts must
already be in their final order before grouping, and ``first_seen_step`` is derived from
the category order *before* the merge.
"""

import re
from dataclasses import fields, replace
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset

StepSchedule = List[int]
StepScheduleLike = Union[str, Sequence[int]]

_SCHEDULE_TOKEN_PATTERN = re.compile(r"^\s*(\d+)\s*(?:[x×*]\s*(\d+))?\s*$")


class StepScheduledConceptsDataset(ConceptsDataset):
    """A :class:`ConceptsDataset` whose train concepts are grouped into scheduled steps.

    Carries the schedule metadata needed to interpret the resulting ``T x N`` metric
    matrix, so that callbacks and metrics do not have to be wired up by hand:

    * :meth:`schedule` -- class counts per training step, e.g. ``[14, 1]``.
    * :meth:`category_order` -- category order *before* grouping (length ``N``).
    * :meth:`first_seen_step` -- category name -> index of the training step that first
      includes it. This is what the schedule-aware metrics consume.
    """

    def __init__(
        self,
        name: str,
        train_concepts: List[Concept],
        test_concepts: List[Concept],
        schedule: StepSchedule,
        category_order: Sequence[str],
        group_test: bool = False,
    ):
        super().__init__(name=name, train_concepts=train_concepts, test_concepts=test_concepts)
        self._schedule = list(schedule)
        self._category_order = list(category_order)
        self._group_test = bool(group_test)
        self._first_seen_step = dict(
            zip(self._category_order, compute_first_seen_step(self._category_order, self._schedule))
        )

    def schedule(self) -> StepSchedule:
        return list(self._schedule)

    def category_order(self) -> List[str]:
        """Category order before grouping, i.e. the training order of the original concepts."""
        return list(self._category_order)

    def first_seen_step(self) -> Dict[str, int]:
        """Map category name -> index of the training step that first includes it.

        :raises ValueError: when ``group_test=True``. The mapping is keyed by the original
            category names, but grouped test concepts are named after their step, so the two
            share no names. Schedule-aware metrics describe the rectangular matrix that only
            ``group_test=False`` produces; with ``group_test=True`` both axes collapse to the
            same T steps and the asymmetry they measure no longer exists.
        """
        if self._group_test:
            raise ValueError(
                f"Dataset {self.name()!r} was grouped with group_test=True, so its test concepts are "
                "grouped steps rather than categories and first_seen_step does not apply. "
                "Schedule-aware metrics need group_test=False."
            )
        return dict(self._first_seen_step)

    def group_test(self) -> bool:
        return self._group_test

    def additional_info(self) -> Dict[str, object]:
        return {
            **super().additional_info(),
            "step_schedule": format_step_schedule(self._schedule),
            "step_schedule_parsed": self.schedule(),
            "group_test": self._group_test,
            "category_order": self.category_order(),
            "train_steps": [concept.name for concept in self.train_concepts()],
            "first_seen_step": self.first_seen_step(),
        }


def parse_step_schedule(spec: StepScheduleLike, total_categories: Optional[int] = None) -> StepSchedule:
    """Parse a step-schedule spec into an explicit list of class counts per step.

    Accepted string forms, tokens separated by ``-``, where ``NxM`` repeats ``N`` ``M``
    times (``×`` and ``*`` are accepted in place of ``x``):

    * ``"14-1"`` -> ``[14, 1]``
    * ``"10-5"`` -> ``[10, 5]``
    * ``"3x5"`` -> ``[3, 3, 3, 3, 3]``
    * ``"10-1x5"`` -> ``[10, 1, 1, 1, 1, 1]``

    A sequence of ints is returned as an equivalent list after validation.

    :param spec: schedule string or explicit sequence of class counts.
    :param total_categories: when given, the schedule must sum to exactly this value.
    :raises ValueError: on an empty, malformed, or non-positive schedule, or when the
        schedule does not sum to ``total_categories``.
    """
    schedule = _parse_schedule_string(spec) if isinstance(spec, str) else [int(value) for value in spec]

    if not schedule:
        raise ValueError(f"Step schedule must contain at least one step, got {spec!r}.")
    if any(step <= 0 for step in schedule):
        raise ValueError(f"Step schedule entries must be positive, got {schedule}.")
    if total_categories is not None and sum(schedule) != total_categories:
        raise ValueError(f"Step schedule {schedule} sums to {sum(schedule)}, expected {total_categories}.")

    return schedule


def format_step_schedule(schedule: Sequence[int]) -> str:
    """Render a parsed schedule back into its compact string form, e.g. ``[10, 1, 1, 1]`` -> ``"10-1x3"``."""
    parts: List[str] = []
    index = 0
    while index < len(schedule):
        run_end = index
        while run_end < len(schedule) and schedule[run_end] == schedule[index]:
            run_end += 1
        run_length = run_end - index
        parts.append(f"{schedule[index]}x{run_length}" if run_length > 1 else str(schedule[index]))
        index = run_end
    return "-".join(parts)


def group_concepts_by_schedule(
    concepts: Sequence[Concept],
    schedule: StepScheduleLike,
    name_prefix: str = "step",
    name_mode: str = "indexed",
    name_separator: str = "+",
) -> List[Concept]:
    """Merge consecutive concepts into one concept per scheduled step.

    Merging concatenates every array-valued field along the batch axis and preserves the
    concrete concept class, so a :class:`~pyclad.vision.data.vision_concept.VisionConcept`
    keeps its ``masks`` aligned with ``data``.

    :param concepts: ordered concepts, one per original category.
    :param schedule: class counts per step; must sum to ``len(concepts)``.
    :param name_mode: ``"indexed"`` names steps ``{name_prefix}_{i}``, ``"joined"`` joins
        the original category names with ``name_separator``.
    :raises ValueError: when the schedule does not match the concept count, on an unknown
        ``name_mode``, or when concepts within a step cannot be merged.
    """
    if name_mode not in {"indexed", "joined"}:
        raise ValueError(f"name_mode must be 'indexed' or 'joined', got {name_mode!r}.")

    parsed = parse_step_schedule(schedule)
    if sum(parsed) != len(concepts):
        raise ValueError(f"Schedule sum ({sum(parsed)}) does not match number of concepts ({len(concepts)}).")

    positions = _first_seen_index_per_position(parsed)
    chunks: List[List[Concept]] = [[] for _ in parsed]
    for position, concept in enumerate(concepts):
        chunks[positions[position]].append(concept)

    grouped: List[Concept] = []
    for step_index, chunk in enumerate(chunks):
        if name_mode == "joined":
            group_name = name_separator.join(concept.name for concept in chunk)
        else:
            group_name = f"{name_prefix}_{step_index}"

        grouped.append(_merge_concepts(chunk, group_name))

    return grouped


def apply_step_schedule(
    dataset: ConceptsDataset,
    schedule: StepScheduleLike,
    group_test: bool = False,
    dataset_name: Optional[str] = None,
    name_mode: str = "indexed",
    name_separator: str = "+",
) -> StepScheduledConceptsDataset:
    """Group a dataset's concepts according to ``schedule``.

    Train concepts are always grouped. Test concepts are kept per category unless
    ``group_test=True``, so that evaluation still reports per-class metrics (the CDAD
    convention) and the metric matrix is ``T x N``.

    The dataset must already be in its final concept order -- grouping merges *consecutive*
    concepts, so any reordering has to happen first.

    :param group_test: when True, test concepts are grouped with the same schedule, which
        requires them to align 1:1 with the train concepts and yields a ``T x T`` matrix.
        That is a different experiment from the ``T x N`` one; do not mix the two.
    :raises ValueError: when the schedule does not sum to the number of train concepts, or
        when ``group_test=True`` and the train/test concepts do not align.
    """
    train_concepts = dataset.train_concepts()
    parsed = parse_step_schedule(schedule, total_categories=len(train_concepts))
    category_order = [concept.name for concept in train_concepts]

    grouped_train = group_concepts_by_schedule(
        concepts=train_concepts,
        schedule=parsed,
        name_mode=name_mode,
        name_separator=name_separator,
    )

    test_concepts = dataset.test_concepts()
    if group_test:
        test_names = [concept.name for concept in test_concepts]
        if test_names != category_order:
            raise ValueError(
                "group_test=True requires test concepts to align 1:1 with train concepts, in the "
                f"same order. Train concepts: {category_order}. Test concepts: {test_names}."
            )
        resolved_test = group_concepts_by_schedule(
            concepts=test_concepts,
            schedule=parsed,
            name_mode=name_mode,
            name_separator=name_separator,
        )
    else:
        resolved_test = list(test_concepts)

    return StepScheduledConceptsDataset(
        name=dataset_name or f"{dataset.name()}__{format_step_schedule(parsed)}",
        train_concepts=grouped_train,
        test_concepts=resolved_test,
        schedule=parsed,
        category_order=category_order,
        group_test=group_test,
    )


def compute_first_seen_step(ordered_concept_names: Sequence[str], schedule: StepScheduleLike) -> List[int]:
    """Return, for each category, the index of the training step that first includes it.

    ``ordered_concept_names`` is the per-category training order (length ``N``) *before*
    grouping. The returned list has the same length and order.

    For MVTec with ``"14-1"`` and alphabetical ordering the result is
    ``[0] * 14 + [1]`` -- everything but ``zipper`` enters training at step 0.

    :raises ValueError: when the schedule does not sum to ``len(ordered_concept_names)``.
    """
    parsed = parse_step_schedule(schedule, total_categories=len(ordered_concept_names))
    return _first_seen_index_per_position(parsed)


def first_seen_step_for_test_order(
    ordered_concept_names: Sequence[str],
    schedule: StepScheduleLike,
    test_order: Sequence[str],
) -> List[int]:
    """Re-align first-seen step indices to an arbitrary evaluation (column) order.

    The callbacks in this library align the mapping themselves, so this helper exists for
    callers that build a concept-level matrix outside pyCLAD and need the same alignment.

    :param ordered_concept_names: training order before grouping.
    :param test_order: column order used by the evaluation callback.
    :raises KeyError: when a name in ``test_order`` is absent from ``ordered_concept_names``.
    """
    name_to_step = dict(zip(ordered_concept_names, compute_first_seen_step(ordered_concept_names, schedule)))

    aligned: List[int] = []
    for name in test_order:
        if name not in name_to_step:
            raise KeyError(
                f"Test concept {name!r} is not present in the training order; its first-seen step is unknown."
            )
        aligned.append(name_to_step[name])
    return aligned


def _first_seen_index_per_position(schedule: StepSchedule) -> List[int]:
    """Map each position in a merged sequence to the step it lands in.

    Shared by :func:`compute_first_seen_step` and :func:`group_concepts_by_schedule` so the two
    cannot disagree about where a schedule's chunk boundaries fall.
    """
    positions: List[int] = []
    for step_index, step_size in enumerate(schedule):
        positions.extend([step_index] * step_size)
    return positions


def _parse_schedule_string(spec: str) -> StepSchedule:
    if not spec.strip():
        raise ValueError("Step schedule string is empty.")

    schedule: StepSchedule = []
    for raw_token in spec.split("-"):
        match = _SCHEDULE_TOKEN_PATTERN.match(raw_token)
        if match is None:
            raise ValueError(
                f"Invalid step-schedule token {raw_token!r} in {spec!r}. Expected 'N' or 'NxM', e.g. '14' or '3x5'."
            )
        size = int(match.group(1))
        repeat = int(match.group(2)) if match.group(2) is not None else 1
        if size <= 0 or repeat <= 0:
            raise ValueError(f"Step-schedule token {raw_token!r} in {spec!r} must use positive numbers.")
        schedule.extend([size] * repeat)
    return schedule


def _merge_concepts(chunk: Sequence[Concept], group_name: str) -> Concept:
    """Merge concepts of one step, preserving their concrete class and array fields.

    Contract: every field of a :class:`Concept` other than ``name`` is either ``None`` or an
    array whose first axis is the batch, so merging concatenates it along that axis. This holds
    for ``Concept`` and :class:`~pyclad.vision.data.vision_concept.VisionConcept`; a subclass
    adding a scalar or per-concept field would need its own merge rule here.
    """
    first = chunk[0]
    if len(chunk) == 1:
        return replace(first, name=group_name)

    concept_type = type(first)
    for concept in chunk[1:]:
        if type(concept) is not concept_type:
            raise ValueError(
                f"Cannot group concepts of different type within one step: "
                f"{concept_type.__name__} and {type(concept).__name__}."
            )

    merged_fields = {}
    for field in fields(concept_type):
        if field.name == "name":
            continue
        values = [getattr(concept, field.name) for concept in chunk]
        if all(value is None for value in values):
            merged_fields[field.name] = None
        elif any(value is None for value in values):
            raise ValueError(
                f"Cannot group concepts where some define {field.name!r} and others do not: "
                f"{[concept.name for concept in chunk]}."
            )
        else:
            merged_fields[field.name] = np.concatenate([np.asarray(value) for value in values], axis=0)

    return concept_type(name=group_name, **merged_fields)
