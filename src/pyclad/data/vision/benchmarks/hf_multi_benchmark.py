from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

from datasets import load_dataset

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.vision_reader import (
    VisionSample,
    build_concepts_dataset_from_samples,
    validate_read_options,
)
from pyclad.data.vision.benchmarks.registry import resolve_vision_benchmark_root

INCLAD_MD_HF_REPO = "anonmllab/inclad-bench"
INCLAD_MD_HF_CONFIG = "inclad-md"

INCLAD_MD_ORDERINGS = ("easy_to_hard", "hard_to_easy")

InCLADMDOrdering = Literal["easy_to_hard", "hard_to_easy"]

INCLAD_MD_SOURCE_DATASETS = ("btech", "dagm", "mpdd", "mvtec", "visa")


def _resolve_roots(
    roots: Optional[dict[str, Union[str, Path]]],
    registry_path: Optional[Union[str, Path]],
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for benchmark in INCLAD_MD_SOURCE_DATASETS:
        explicit = roots.get(benchmark) if roots else None
        resolved[benchmark] = resolve_vision_benchmark_root(
            benchmark=benchmark,
            root=explicit,
            registry_path=registry_path,
        )
    return resolved


def _parse_hf_manifest(hf_dataset) -> list[dict[str, str]]:
    buf = io.StringIO()
    df = hf_dataset.to_pandas()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return list(csv.DictReader(buf))


def _index_samples(
    rows: list[dict[str, str]],
    source_roots: dict[str, Path],
    categories: Optional[Sequence[str]],
    max_train_per_category: Optional[int],
    max_test_per_category: Optional[int],
) -> list[VisionSample]:
    seen: dict[str, int] = {}
    for row in rows:
        cat = row["category"]
        if cat not in seen:
            try:
                seen[cat] = int(row["category_order"])
            except (ValueError, KeyError):
                seen[cat] = len(seen) + 1

    ordered_categories = [cat for cat, _ in sorted(seen.items(), key=lambda item: item[1])]

    if categories is not None:
        missing = [c for c in categories if c not in seen]
        if missing:
            raise ValueError(
                f"Requested categories not found in InCLAD-MD manifest: {missing}. " f"Available: {ordered_categories}"
            )
        ordered_categories = list(categories)

    selected_set = set(ordered_categories)

    samples_by_key: dict[tuple[str, str], list[VisionSample]] = {}
    for row in rows:
        category = row["category"]
        if category not in selected_set:
            continue

        split_raw = row.get("split", "").strip()
        if split_raw not in ("train", "test"):
            continue

        source_dataset = row["source_dataset"].strip().lower()
        root = source_roots.get(source_dataset)
        if root is None:
            raise ValueError(
                f"No root resolved for source dataset '{source_dataset}'. "
                f"Pass it in the roots= mapping or set PYCLAD_VISION_DATASETS_ROOT."
            )

        image_relpath = row.get("image_relpath", "").strip()
        if not image_relpath:
            continue
        image_path = root / image_relpath
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}\n" f"  (source_dataset={source_dataset}, root={root})"
            )

        mask_relpath = row.get("mask_relpath", "").strip()
        mask_path: Optional[Path] = None
        if mask_relpath:
            candidate = root / mask_relpath
            if candidate.exists():
                mask_path = candidate

        try:
            image_label = int(row.get("image_label", "0"))
        except ValueError:
            image_label = 0

        defect_type = row.get("defect_type", "").strip() or None

        sample = VisionSample(
            category=category,
            split=split_raw,
            image_path=image_path,
            image_label=image_label,
            mask_path=mask_path,
            defect_type=defect_type,
        )
        samples_by_key.setdefault((category, split_raw), []).append(sample)

    samples: list[VisionSample] = []
    for category in ordered_categories:
        train = samples_by_key.get((category, "train"), [])
        test = samples_by_key.get((category, "test"), [])
        if max_train_per_category is not None:
            train = train[:max_train_per_category]
        if max_test_per_category is not None:
            test = test[:max_test_per_category]
        samples.extend(train)
        samples.extend(test)

    return samples


class InCLADMDDataset(ConceptsDataset):
    """
    Loader for the InCLAD-MD multi-dataset benchmark hosted on HuggingFace
    as the ``"inclad-md"`` config of ``anonmllab/inclad-bench``.

    InCLAD-MD is a single sequential benchmark built from selected categories
    across all five InCLAD-Bench source datasets (BTech, DAGM, MPDD, MVTec AD,
    VisA). Categories are prefixed with their source dataset name, e.g.
    ``mvtec__bottle``, ``dagm__Class4``.

    Because images come from multiple source datasets, a separate local root
    must be available for each one. Roots are resolved via the ``roots``
    parameter, per-benchmark env vars, ``PYCLAD_VISION_DATASETS_ROOT``, or
    the pyCLAD vision dataset registry — same as :class:`InCLADBenchDataset`.
    """

    def __init__(
        self,
        ordering: InCLADMDOrdering = "easy_to_hard",
        roots: Optional[dict[str, Union[str, Path]]] = None,
        registry_path: Optional[Union[str, Path]] = None,
        categories: Optional[Sequence[str]] = None,
        data_mode: str = "numpy",
        resize_to: Optional[tuple[int, int]] = None,
        color_mode: str = "rgb",
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        :param ordering: ``"easy_to_hard"`` or ``"hard_to_easy"``.
        :param roots: Mapping ``{benchmark_name: path}`` for source dataset roots.
            Missing entries are resolved via env vars / registry.
        :param registry_path: Optional path to the vision dataset registry JSON.
        :param categories: Subset of categories (prefixed, e.g. ``"mvtec__bottle"``).
            ``None`` = all.
        :param data_mode: ``"numpy"`` or ``"paths"``.
        :param resize_to: ``(height, width)`` to resize images. ``None`` = keep original.
        :param color_mode: ``"rgb"`` or ``"grayscale"``.
        :param max_train_samples_per_category: Cap on training samples per category.
        :param max_test_samples_per_category: Cap on test samples per category.
        :param cache_dir: HuggingFace cache directory.
        """
        if ordering not in INCLAD_MD_ORDERINGS:
            raise ValueError(f"Unknown ordering '{ordering}'. Available: {INCLAD_MD_ORDERINGS}")
        validate_read_options(data_mode=data_mode, color_mode=color_mode)

        source_roots = _resolve_roots(roots, registry_path)

        hf_dataset = load_dataset(INCLAD_MD_HF_REPO, INCLAD_MD_HF_CONFIG, split=ordering, cache_dir=cache_dir)
        rows = _parse_hf_manifest(hf_dataset)

        samples = _index_samples(
            rows=rows,
            source_roots=source_roots,
            categories=categories,
            max_train_per_category=max_train_samples_per_category,
            max_test_per_category=max_test_samples_per_category,
        )

        dataset = build_concepts_dataset_from_samples(
            samples=samples,
            categories=categories,
            data_mode=data_mode,
            resize_to=resize_to,
            color_mode=color_mode,
            dataset_name=f"InCLAD-MD-{ordering}",
        )

        super().__init__(
            name=dataset.name(),
            train_concepts=dataset.train_concepts(),
            test_concepts=dataset.test_concepts(),
        )


def load_inclad_md(
    ordering: InCLADMDOrdering = "easy_to_hard",
    roots: Optional[dict[str, Union[str, Path]]] = None,
    registry_path: Optional[Union[str, Path]] = None,
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> ConceptsDataset:
    """Convenience function — see :class:`InCLADMDDataset` for parameter docs."""
    return InCLADMDDataset(
        ordering=ordering,
        roots=roots,
        registry_path=registry_path,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
        cache_dir=cache_dir,
    )
