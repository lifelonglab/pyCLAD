from __future__ import annotations

import csv
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Sequence, Tuple, Union

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.vision.data.base import (
    ImageLoadOptions,
    build_concepts_dataset_from_samples,
    select_categories,
)
from pyclad.vision.data.benchmarks.manifest import (
    VISION_BENCHMARK_MANIFEST_FIELDNAMES,
    build_vision_benchmark_manifest_spec,
)
from pyclad.vision.data.benchmarks.readers import build_vision_benchmark_reader

if TYPE_CHECKING:
    import pandas as pd

INCLAD_BENCH_HF_REPO = "lifelonglab/inclad-benchmark"

# Config -> the orderings published for it; each ordering is one split of the config on the Hub.
INCLAD_BENCH_ORDERINGS = {
    "btech": ("easy_to_hard", "hard_to_easy", "random"),
    "dagm": ("easy_to_hard", "hard_to_easy", "random"),
    "inclad-md": ("easy_to_hard", "hard_to_easy"),
    "mpdd": ("easy_to_hard", "hard_to_easy", "random"),
    "mvtec": ("easy_to_hard", "hard_to_easy", "random"),
    "visa": ("easy_to_hard", "hard_to_easy", "random"),
}

INCLAD_BENCH_CONFIGS = tuple(INCLAD_BENCH_ORDERINGS)

InCLADBenchConfig = Literal["btech", "dagm", "inclad-md", "mpdd", "mvtec", "visa"]
InCLADBenchOrdering = Literal["easy_to_hard", "hard_to_easy", "random"]

MULTIDATASET_NAME_SEPARATOR = "__"


class InCLADBenchDataset(ConceptsDataset):
    """Loader for the InCLAD benchmark hosted on Hugging Face
    (https://huggingface.co/datasets/lifelonglab/inclad-benchmark).

    The Hub holds only sample manifests: image paths, labels, masks and the category order of
    every ordering. The images are not on the Hub; ``roots`` points at the local copies.

    Five configs (``btech``, ``dagm``, ``mpdd``, ``mvtec``, ``visa``) are streams over the
    categories of one source dataset. ``inclad-md`` is one stream of categories drawn from all five,
    named ``<source>__<category>``. Every manifest row names its ``source_dataset`` and a path
    relative to that dataset's folder, so all six configs load the same way: each row is read from
    ``roots[source_dataset]``.
    """

    def __init__(
        self,
        benchmark: InCLADBenchConfig,
        roots: Mapping[str, Union[str, Path]],
        ordering: InCLADBenchOrdering = "easy_to_hard",
        categories: Optional[Sequence[str]] = None,
        data_mode: str = "numpy",
        resize_to: Optional[tuple[int, int]] = None,
        color_mode: str = "rgb",
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        :param benchmark: one of ``"btech"``, ``"dagm"``, ``"inclad-md"``, ``"mpdd"``, ``"mvtec"``,
            ``"visa"``.
        :param roots: source dataset name -> local folder with its images, e.g.
            ``{"mvtec": "/data/mvtec_ad"}``. ``inclad-md`` needs all five sources. Entries for
            sources a config does not use are ignored, so one mapping serves every config.
        :param ordering: ``"easy_to_hard"``, ``"hard_to_easy"`` or ``"random"``; ``inclad-md`` has
            no ``"random"``.
        :param categories: subset of concept names (``"mvtec__bottle"`` for ``inclad-md``), which
            also fixes their order. ``None`` takes every category in the order of ``ordering``.
        :param data_mode: ``"numpy"`` to load images as arrays, ``"paths"`` for file paths only.
        :param resize_to: ``(height, width)`` to resize images. ``None`` keeps original size.
        :param color_mode: ``"rgb"`` or ``"grayscale"``.
        :param max_train_samples_per_category: cap on training samples per category.
        :param max_test_samples_per_category: cap on test samples per category.
        :param cache_dir: Hugging Face cache directory. ``None`` uses the default.
        """
        config = benchmark.lower()
        if config not in INCLAD_BENCH_ORDERINGS:
            raise ValueError(f"Unknown InCLAD benchmark config '{benchmark}'. Available: {INCLAD_BENCH_CONFIGS}")
        if ordering not in INCLAD_BENCH_ORDERINGS[config]:
            raise ValueError(
                f"Unknown ordering '{ordering}' for config '{config}'. Available: {INCLAD_BENCH_ORDERINGS[config]}"
            )
        options = ImageLoadOptions(data_mode=data_mode, resize_to=resize_to, color_mode=color_mode)

        manifest = _download_manifest(config=config, ordering=ordering, cache_dir=cache_dir)
        selected_categories = select_categories(_category_order(manifest), categories)
        selected_rows = manifest[manifest["category"].isin(selected_categories)]
        source_roots = _resolve_roots(config=config, roots=roots, sources=selected_rows["source_dataset"].unique())

        samples = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for source, source_rows in selected_rows.groupby("source_dataset", sort=False):
                csv_path = _write_manifest_csv(source_rows, Path(tmp_dir) / f"{source}.csv")
                reader = build_vision_benchmark_reader(
                    root=source_roots[source],
                    benchmark=build_vision_benchmark_manifest_spec(benchmark=source, csv_path=csv_path),
                )
                samples.extend(
                    reader.index_samples(
                        categories=[c for c in selected_categories if c in set(source_rows["category"])],
                        max_train_samples_per_category=max_train_samples_per_category,
                        max_test_samples_per_category=max_test_samples_per_category,
                    )
                )

        dataset = build_concepts_dataset_from_samples(
            samples=samples,
            dataset_name=f"InCLAD-{config}-{ordering}",
            categories=selected_categories,
            options=options,
        )
        super().__init__(
            name=dataset.name(),
            train_concepts=dataset.train_concepts(),
            test_concepts=dataset.test_concepts(),
        )


def load_inclad_bench(
    benchmark: InCLADBenchConfig,
    roots: Mapping[str, Union[str, Path]],
    ordering: InCLADBenchOrdering = "easy_to_hard",
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> ConceptsDataset:
    return InCLADBenchDataset(
        benchmark=benchmark,
        roots=roots,
        ordering=ordering,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
        cache_dir=cache_dir,
    )


def split_multi_dataset_concept_name(
    concept_name: str, separator: str = MULTIDATASET_NAME_SEPARATOR
) -> Tuple[str, str]:
    if separator not in concept_name:
        raise ValueError(f"Concept name {concept_name!r} does not contain the separator {separator!r}.")
    source, category = concept_name.split(separator, 1)
    return source, category


def _download_manifest(config: str, ordering: str, cache_dir: Optional[str]) -> "pd.DataFrame":
    from datasets import Features, Value, load_dataset

    features = Features({name: Value("string") for name in VISION_BENCHMARK_MANIFEST_FIELDNAMES})
    hf_dataset = load_dataset(
        INCLAD_BENCH_HF_REPO,
        config,
        split=ordering,
        cache_dir=cache_dir,
        features=features,
        keep_default_na=False,
    )
    return hf_dataset.to_pandas()


def _category_order(manifest: "pd.DataFrame") -> list[str]:
    positions: dict[str, int] = {}
    for category, raw_position in zip(manifest["category"], manifest["category_order"]):
        position = int(raw_position)
        if positions.setdefault(category, position) != position:
            raise ValueError(f"Category {category!r} has more than one category_order value in the manifest.")

    counts = Counter(positions.values())
    tied = sorted(category for category, position in positions.items() if counts[position] > 1)
    if tied:
        raise ValueError(f"Categories {tied} share a category_order value, so their order is undefined.")
    return sorted(positions, key=positions.__getitem__)


def _resolve_roots(config: str, roots: Mapping[str, Union[str, Path]], sources: Sequence[str]) -> dict[str, Path]:
    missing = [source for source in sources if source not in roots]
    if missing:
        raise ValueError(
            f"InCLAD config '{config}' reads images of {missing}, but roots= has no entry for them. "
            f"Given roots: {sorted(roots)}."
        )

    resolved = {source: Path(roots[source]).expanduser().resolve() for source in sources}
    for source, root in resolved.items():
        if not root.exists():
            raise FileNotFoundError(f"Root for InCLAD source dataset '{source}' does not exist: {root}")
    return resolved


def _write_manifest_csv(rows: "pd.DataFrame", output_path: Path) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VISION_BENCHMARK_MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows[list(VISION_BENCHMARK_MANIFEST_FIELDNAMES)].to_dict("records"))
    return output_path
