"""Multidataset: one concept stream drawn from several source datasets.

A multidataset is described by an ordered list of :class:`DatasetBlock`, each contributing
its categories under a ``<alias>__<category>`` name. Unlike a single-source dataset there
is no shared root: every block resolves its images against its own root, so the same
relative path in two sources refers to two different files.

Blocks delegate to the existing per-dataset readers, so every layout that
:func:`~pyclad.vision.data.benchmarks.readers.build_vision_benchmark_reader` understands --
including a hand-written :class:`FolderBenchmarkSpec` or :class:`CsvBenchmarkSpec` -- is a
valid block without any extra code.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.vision.data.base import VisionBenchmarkReader
from pyclad.vision.data.benchmarks.readers import (
    CsvBenchmarkSpec,
    FolderBenchmarkSpec,
    VisionBenchmarkSpec,
    build_vision_benchmark_reader,
)
from pyclad.vision.data.sample import VisionSample

DEFAULT_NAME_SEPARATOR = "__"


@dataclass(frozen=True)
class DatasetBlock:
    """One source dataset's contribution to a multidataset.

    :param dataset: source dataset key (``"mvtec"``, ``"visa"``, ...) or an explicit
        :class:`FolderBenchmarkSpec` / :class:`CsvBenchmarkSpec` for a custom layout.
    :param root: this block's dataset root. When omitted it is looked up in the ``roots``
        mapping passed to :func:`read_multi_dataset`, keyed by the source dataset name.
    :param categories: categories to take, in order. ``None`` takes every category the
        source reader reports, in its own order.
    :param alias: name used to prefix this block's concepts; defaults to the dataset name.
    """

    dataset: Union[str, VisionBenchmarkSpec]
    root: Optional[Union[str, Path]] = None
    categories: Optional[Sequence[str]] = None
    alias: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", None if self.root is None else str(self.root))
        if self.categories is not None:
            object.__setattr__(self, "categories", tuple(self.categories))

    def dataset_name(self) -> str:
        """Name of the source dataset, used to look this block's root up in ``roots``."""
        return self.dataset if isinstance(self.dataset, str) else self.dataset.name

    def prefix(self) -> str:
        return self.alias or self.dataset_name()

    def concept_names(self, separator: str = DEFAULT_NAME_SEPARATOR) -> List[str]:
        if self.categories is None:
            raise ValueError(
                f"Block {self.prefix()!r} does not list its categories, so its concept names are "
                "only known after reading its root. List them explicitly to use this."
            )
        return [f"{self.prefix()}{separator}{category}" for category in self.categories]

    def reversed(self) -> "DatasetBlock":
        if self.categories is None:
            raise ValueError(
                f"Block {self.prefix()!r} does not list its categories, so it cannot be reversed. "
                "List them explicitly to use this."
            )
        return DatasetBlock(
            dataset=self.dataset,
            root=self.root,
            categories=tuple(reversed(self.categories)),
            alias=self.alias,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"dataset": _benchmark_to_payload(self.dataset)}
        if self.root is not None:
            payload["root"] = self.root
        if self.categories is not None:
            payload["categories"] = list(self.categories)
        if self.alias is not None:
            payload["alias"] = self.alias
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetBlock":
        return cls(
            dataset=_benchmark_from_payload(payload["dataset"]),
            root=payload.get("root"),
            categories=payload.get("categories"),
            alias=payload.get("alias"),
        )


@dataclass(frozen=True)
class MultiDatasetSpec:
    """An ordered, reproducible description of a multidataset concept stream.

    Block order is the order of ``blocks``; category order within a block is the order of
    that block's ``categories``. Both survive all the way into
    :meth:`~pyclad.data.datasets.concepts_dataset.ConceptsDataset.train_concepts`, so this
    object alone determines the continual sequence.
    """

    name: str
    blocks: Sequence[DatasetBlock] = field(default_factory=tuple)
    name_separator: str = DEFAULT_NAME_SEPARATOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if not self.blocks:
            raise ValueError(f"Multidataset {self.name!r} must contain at least one block.")

        seen: Dict[str, str] = {}
        for block in self.blocks:
            if block.categories is None:
                continue
            for concept_name in block.concept_names(self.name_separator):
                if concept_name in seen:
                    raise ValueError(
                        f"Duplicate concept name {concept_name!r} in multidataset {self.name!r}. "
                        "Give one of the blocks a distinct alias."
                    )
                seen[concept_name] = block.prefix()

    def category_order(self) -> List[str]:
        """Full concept sequence, prefixed and in stream order."""
        order: List[str] = []
        for block in self.blocks:
            order.extend(block.concept_names(self.name_separator))
        return order

    def reversed(self) -> "MultiDatasetSpec":
        """Reverse the stream at both levels: block order and category order within blocks."""
        return MultiDatasetSpec(
            name=f"{self.name}_reversed",
            blocks=tuple(block.reversed() for block in reversed(self.blocks)),
            name_separator=self.name_separator,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "name_separator": self.name_separator,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MultiDatasetSpec":
        return cls(
            name=payload["name"],
            blocks=[DatasetBlock.from_dict(block) for block in payload["blocks"]],
            name_separator=payload.get("name_separator", DEFAULT_NAME_SEPARATOR),
        )

    def to_json(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        return target

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "MultiDatasetSpec":
        return cls.from_dict(json.loads(Path(path).read_text()))


class MultiDatasetReader(VisionBenchmarkReader):
    """Reads a multidataset by delegating each block to its own source-dataset reader.

    Samples are emitted in block order, and each block's samples in its category order, which
    is what fixes the concept order downstream: ``build_concepts_dataset_from_samples`` infers
    the order from sample occurrence when no explicit category list is given.
    """

    def __init__(self, spec: MultiDatasetSpec, roots: Optional[Mapping[str, Union[str, Path]]] = None):
        super().__init__(root=None, name=spec.name)
        self.spec = spec
        self._roots = {key: Path(value) for key, value in (roots or {}).items()}

    def available_categories(self) -> List[str]:
        names: List[str] = []
        for block in self.spec.blocks:
            reader = self._reader_for(block)
            categories = block.categories if block.categories is not None else reader.available_categories()
            names.extend(f"{block.prefix()}{self.spec.name_separator}{category}" for category in categories)
        return names

    def index_samples(
        self,
        categories: Optional[Sequence[str]] = None,
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
    ) -> List[VisionSample]:
        samples: List[VisionSample] = []
        for block in self.spec.blocks:
            reader = self._reader_for(block)
            try:
                block_samples = reader.index_samples(
                    categories=block.categories,
                    max_train_samples_per_category=max_train_samples_per_category,
                    max_test_samples_per_category=max_test_samples_per_category,
                )
            except ValueError as error:
                raise ValueError(f"Multidataset block {block.prefix()!r}: {error}") from error

            prefix = f"{block.prefix()}{self.spec.name_separator}"
            samples.extend(
                VisionSample(
                    category=f"{prefix}{sample.category}",
                    split=sample.split,
                    image_path=sample.image_path,
                    image_label=sample.image_label,
                    mask_path=sample.mask_path,
                    defect_type=sample.defect_type,
                )
                for sample in block_samples
            )

        if categories is not None:
            requested = set(categories)
            samples = [sample for sample in samples if sample.category in requested]
        return samples

    def _reader_for(self, block: DatasetBlock) -> VisionBenchmarkReader:
        return build_vision_benchmark_reader(root=self._resolve_root(block), benchmark=block.dataset)

    def _resolve_root(self, block: DatasetBlock) -> Path:
        if block.root is not None:
            return Path(block.root)

        dataset_name = block.dataset_name()
        if dataset_name in self._roots:
            return self._roots[dataset_name]

        raise ValueError(
            f"No root for multidataset block {block.prefix()!r} (source dataset {dataset_name!r}). "
            f"Set it on the block, or pass roots={{{dataset_name!r}: ...}}. "
            f"Known roots: {sorted(self._roots)}."
        )


def read_multi_dataset(
    spec: Optional[MultiDatasetSpec] = None,
    roots: Optional[Mapping[str, Union[str, Path]]] = None,
    dataset_name: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[Tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> ConceptsDataset:
    """Read a multidataset into a :class:`ConceptsDataset`, one concept per source category.

    Defaults to the published :data:`INCLAD_MD_SPEC`. Test concepts carrying ground-truth
    masks come back as :class:`~pyclad.vision.data.vision_concept.VisionConcept`, with masks
    resolved against each block's own root.

    :param roots: source dataset name -> root, used for blocks that do not set ``root``.
    :param categories: optional subset of prefixed concept names, e.g. ``["mvtec__bottle"]``,
        which also fixes their order.
    """
    resolved_spec = INCLAD_MD_SPEC if spec is None else spec
    reader = MultiDatasetReader(spec=resolved_spec, roots=roots)
    return reader.read_dataset(
        dataset_name=dataset_name or resolved_spec.name,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def split_multi_dataset_concept_name(concept_name: str, separator: str = DEFAULT_NAME_SEPARATOR) -> Tuple[str, str]:
    """Split ``"mvtec__bottle"`` into ``("mvtec", "bottle")``, for per-source reporting.

    Splits on the first separator, so underscores inside a category name survive.
    """
    if separator not in concept_name:
        raise ValueError(f"Concept name {concept_name!r} does not contain the separator {separator!r}.")
    source, category = concept_name.split(separator, 1)
    return source, category

INCLAD_MD_SOURCE_DATASETS = ("btech", "dagm", "mpdd", "mvtec", "visa")
INCLAD_MD_ORDERINGS = ("easy_to_hard", "hard_to_easy")

INCLAD_MD_SPEC = MultiDatasetSpec(
    name="inclad-md",
    blocks=(
        DatasetBlock(dataset="btech", categories=("03", "01", "02")),
        DatasetBlock(
            dataset="mpdd",
            categories=("connector", "bracket_brown", "metal_plate", "bracket_white", "tubes", "bracket_black"),
        ),
        DatasetBlock(dataset="dagm", categories=("Class4", "Class9", "Class7", "Class3", "Class5", "Class8")),
        DatasetBlock(dataset="visa", categories=("pcb4", "chewinggum", "fryum", "pcb2", "macaroni2", "capsules")),
        DatasetBlock(dataset="mvtec", categories=("bottle", "leather", "toothbrush", "carpet", "screw", "pill")),
    ),
)


def load_inclad_md(
    roots: Mapping[str, Union[str, Path]],
    ordering: str = "easy_to_hard",
    **kwargs: Any,
) -> ConceptsDataset:
    """Load the published InCLAD-MD benchmark.

    :param roots: source dataset name -> root, e.g. ``{"mvtec": "/data/mvtec_ad", ...}``.
    :param ordering: ``"easy_to_hard"`` or ``"hard_to_easy"``. Unlike InCLAD-Bench, InCLAD-MD
        has no ``"random"`` ordering.
    """
    if ordering not in INCLAD_MD_ORDERINGS:
        raise ValueError(
            f"Unsupported InCLAD-MD ordering {ordering!r}. Available: {list(INCLAD_MD_ORDERINGS)}. "
            "Note that InCLAD-MD has no 'random' ordering, unlike InCLAD-Bench."
        )

    spec = INCLAD_MD_SPEC if ordering == "easy_to_hard" else INCLAD_MD_SPEC.reversed()
    return read_multi_dataset(spec=spec, roots=roots, dataset_name=f"inclad-md-{ordering}", **kwargs)


_BENCHMARK_SPEC_TYPES = {"folder": FolderBenchmarkSpec, "csv": CsvBenchmarkSpec}
_TUPLE_SPEC_FIELDS = {"image_extensions"}


def _benchmark_to_payload(benchmark: Union[str, VisionBenchmarkSpec]) -> Union[str, Dict[str, Any]]:
    if isinstance(benchmark, str):
        return benchmark
    kind = "folder" if isinstance(benchmark, FolderBenchmarkSpec) else "csv"
    return {"kind": kind, **asdict(benchmark)}


def _benchmark_from_payload(payload: Union[str, Mapping[str, Any]]) -> Union[str, VisionBenchmarkSpec]:
    if isinstance(payload, str):
        return payload

    values = dict(payload)
    spec_type = _BENCHMARK_SPEC_TYPES[values.pop("kind")]
    known = {f.name for f in fields(spec_type)}
    return spec_type(
        **{key: tuple(value) if key in _TUPLE_SPEC_FIELDS else value for key, value in values.items() if key in known}
    )
