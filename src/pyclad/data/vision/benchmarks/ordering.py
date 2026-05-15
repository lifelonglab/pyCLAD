from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.vision._utils import find_duplicates
from pyclad.data.vision.benchmarks.readers import (
    PREDEFINED_BENCHMARK_ALIASES,
    available_vision_benchmarks,
)
from pyclad.data.vision.benchmarks.registry import (
    read_registered_vision_benchmark_dataset,
)

ORDERING_MODE_PATTERN = re.compile(r"_(easy_to_hard|hard_to_easy)(?:_|$)")
SERIALIZED_ORDER_COLUMNS = ("easy_to_hard", "hard_to_easy", "concept_order", "concepts", "order")


def infer_vision_benchmark_from_ordering_path(ordering_path: Union[str, Path]) -> str:
    path = Path(ordering_path)
    stem = path.stem.lower()
    candidate_names = sorted(
        set(available_vision_benchmarks()) | set(PREDEFINED_BENCHMARK_ALIASES),
        key=len,
        reverse=True,
    )

    for candidate in candidate_names:
        if stem == candidate or stem.startswith(f"{candidate}_"):
            return PREDEFINED_BENCHMARK_ALIASES.get(candidate, candidate)

    raise ValueError(
        f"Could not infer vision benchmark from ordering filename '{path.name}'. " "Pass benchmark explicitly."
    )


def infer_ordering_mode_from_path(ordering_path: Union[str, Path]) -> Optional[str]:
    match = ORDERING_MODE_PATTERN.search(Path(ordering_path).stem.lower())
    if match is None:
        return None
    return match.group(1)


def resolve_benchmark_ordering_path(
    ordering_dir: Union[str, Path],
    benchmark: str,
    ordering_mode: str,
    ordering_metric: str = "roc-auc",
) -> Path:
    resolved_dir = Path(ordering_dir).expanduser().resolve()
    normalized_benchmark = PREDEFINED_BENCHMARK_ALIASES.get(str(benchmark).lower(), str(benchmark).lower())

    exact_filename = f"{normalized_benchmark}_{ordering_mode}_{ordering_metric}.csv"
    exact_path = resolved_dir / exact_filename
    if exact_path.exists():
        return exact_path

    wildcard_pattern = f"{normalized_benchmark}_{ordering_mode}_*_{ordering_metric}.csv"
    matches = sorted(resolved_dir.glob(wildcard_pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous benchmark ordering in {resolved_dir} for benchmark '{normalized_benchmark}' "
            f"and mode '{ordering_mode}'. Matches: {[match.name for match in matches]}"
        )

    raise FileNotFoundError(
        f"Benchmark ordering file does not exist in {resolved_dir} for benchmark '{normalized_benchmark}', "
        f"mode '{ordering_mode}', and metric '{ordering_metric}'. Expected '{exact_filename}' or a unique match "
        f"for '{wildcard_pattern}'."
    )


def load_concept_order_from_file(
    ordering_path: Union[str, Path],
    benchmark: Optional[str] = None,
    ordering_column: str = "concept",
    ordering_mode: Optional[str] = None,
) -> list[str]:
    path = Path(ordering_path)
    if not path.exists():
        raise FileNotFoundError(f"Ordering file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_concept_order_from_csv(
            ordering_path=path,
            benchmark=benchmark,
            ordering_column=ordering_column,
            ordering_mode=ordering_mode,
        )
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _validate_loaded_concepts(
            _extract_order_from_payload(
                payload=payload,
                benchmark=benchmark,
                ordering_mode=ordering_mode or infer_ordering_mode_from_path(path),
            )
        )
    raise ValueError(f"Unsupported ordering file format: {path}")


def load_concept_order_from_csv(
    ordering_path: Union[str, Path],
    benchmark: Optional[str] = None,
    ordering_column: str = "concept",
    ordering_mode: Optional[str] = None,
) -> list[str]:
    path = Path(ordering_path)
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"Ordering CSV has no header: {path}")
        rows = list(reader)

    if ordering_column in reader.fieldnames:
        concepts = [row[ordering_column].strip() for row in rows if row.get(ordering_column, "").strip()]
        return _validate_loaded_concepts(concepts)

    preferred_columns: list[str] = []
    resolved_mode = ordering_mode or infer_ordering_mode_from_path(path)
    if resolved_mode is not None:
        preferred_columns.append(resolved_mode)
    preferred_columns.extend(column for column in SERIALIZED_ORDER_COLUMNS if column not in preferred_columns)

    filtered_rows = rows
    if benchmark is not None and "benchmark" in reader.fieldnames:
        filtered_rows = [row for row in rows if row.get("benchmark", "").strip().lower() == benchmark.lower()]
        if not filtered_rows:
            raise ValueError(f"Ordering CSV '{path}' does not contain a row for benchmark '{benchmark}'.")

    for column in preferred_columns:
        if column not in reader.fieldnames:
            continue
        for row in filtered_rows:
            raw_value = row.get(column, "")
            if raw_value.strip():
                return _validate_loaded_concepts(_parse_serialized_order(raw_value))

    raise ValueError(
        f"Could not extract concept ordering from '{path}'. "
        f"Expected column '{ordering_column}' or one of {list(preferred_columns)}. "
        f"Available columns: {reader.fieldnames}"
    )


def read_ordered_registered_vision_benchmark_dataset(
    ordering_path: Union[str, Path],
    benchmark: Optional[str] = None,
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
    dataset_name: Optional[str] = None,
    ordering_column: str = "concept",
    ordering_mode: Optional[str] = None,
    data_mode: str = "numpy",
    resize_to: Optional[tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> ConceptsDataset:
    resolved_ordering_path = Path(ordering_path)
    resolved_benchmark = benchmark or infer_vision_benchmark_from_ordering_path(resolved_ordering_path)
    concept_order = load_concept_order_from_file(
        ordering_path=resolved_ordering_path,
        benchmark=resolved_benchmark,
        ordering_column=ordering_column,
        ordering_mode=ordering_mode,
    )
    resolved_dataset_name = dataset_name or resolved_ordering_path.stem

    return read_registered_vision_benchmark_dataset(
        benchmark=resolved_benchmark,
        root=root,
        registry_path=registry_path,
        dataset_name=resolved_dataset_name,
        categories=concept_order,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def _extract_order_from_payload(
    payload: Any,
    benchmark: Optional[str],
    ordering_mode: Optional[str],
) -> list[str]:
    if isinstance(payload, list):
        return _parse_serialized_order(payload)

    if not isinstance(payload, dict):
        raise ValueError("Ordering JSON must contain a list or object with a concept order.")

    for key in ("concept_order", "concepts", "order"):
        if key in payload:
            return _parse_serialized_order(payload[key])

    if ordering_mode and ordering_mode in payload:
        return _parse_serialized_order(payload[ordering_mode])

    if benchmark is not None:
        benchmark_keys = (benchmark, benchmark.lower(), benchmark.upper())
        for key in benchmark_keys:
            if key in payload:
                return _extract_order_from_payload(payload[key], benchmark=benchmark, ordering_mode=ordering_mode)

    raise ValueError(
        "Could not find a concept order in the JSON payload. "
        "Expected one of: concept_order, concepts, order, ordering mode key, or benchmark key."
    )


def _parse_serialized_order(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        raise ValueError("Ordering value is empty.")
    if not isinstance(value, str):
        raise ValueError(f"Unsupported ordering payload type: {type(value)!r}")

    stripped = value.strip()
    if not stripped:
        raise ValueError("Ordering value is empty.")

    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("Serialized ordering JSON must decode to a list.")
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [item.strip() for item in stripped.split(",") if item.strip()]


def _validate_loaded_concepts(concepts: Sequence[str]) -> list[str]:
    validated = [str(concept).strip() for concept in concepts if str(concept).strip()]
    if not validated:
        raise ValueError("Concept order is empty.")

    duplicates = find_duplicates(validated)
    if duplicates:
        raise ValueError(f"Concept order contains duplicates: {duplicates}")

    return validated
