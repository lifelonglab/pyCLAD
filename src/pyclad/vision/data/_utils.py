from __future__ import annotations

from typing import List, Optional, Protocol, Sequence


class _HasCategory(Protocol):
    category: str


def find_duplicates(values: Sequence[str]) -> list[str]:
    """Return values that appear more than once, preserving order of first duplicate occurrence."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def infer_category_order_from_samples(samples: Sequence[_HasCategory]) -> list[str]:
    """Extract unique category names from samples, preserving first-occurrence order."""
    ordered_categories: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        if sample.category not in seen:
            seen.add(sample.category)
            ordered_categories.append(sample.category)
    return ordered_categories


def resolve_category_order(
    samples: Sequence[_HasCategory],
    categories: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return explicit category order if given, otherwise infer from sample occurrence order."""
    if categories is not None:
        return list(categories)
    return infer_category_order_from_samples(samples)
