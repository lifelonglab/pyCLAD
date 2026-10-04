from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.vision.data.base import ImageLoadOptions
from pyclad.vision.data.benchmarks.readers import VisABenchmarkReader
from pyclad.vision.data.lazy_concepts import LazyVisionConceptList
from pyclad.vision.data.sample import VisionSample

CONTINUAL_MEGA_SCENARIOS = (1, 2, 3)
CONTINUAL_MEGA_TASK_SIZES = (5, 10, 30)
CONTINUAL_MEGA_BASE_GROUP = "base"
CONTINUAL_MEGA_IMAGE_SIZE = (336, 336)
CONTINUAL_MEGA_INTERPOLATION = "bicubic"
CONTINUAL_MEGA_ZERO_SHOT_DIRS = {"mvtec": "mvtec_anomaly_detection", "visa": "VisA_20220922"}
CONTINUAL_MEGA_TRAIN_SAMPLES = ("all", "normal")


@dataclass(frozen=True)
class ConceptGroup:
    name: str
    train_samples: List[VisionSample]
    test_concepts: List[Tuple[str, List[VisionSample]]] = field(default_factory=list)
    held_out: bool = False


class ContinualMegaDataset(ConceptsDataset):
    """
    Continual-MEGA benchmark stream: task groups as training concepts, individual classes as test concepts.

    If using, please cite:

    .. code-block:: bibtex

        @article{lee2026continual,
          title={Continual-MEGA: A large-scale benchmark for generalizable continual anomaly detection},
          author={Lee, Geonu and Oh, Yujeong and Jang, Geonhui and Lee, Soyoung and Song, Jeonghyo
                  and Cha, Sungmin and Yoo, YoungJoon},
          journal={Neurocomputing},
          pages={134460},
          year={2026},
          publisher={Elsevier}
        }
    """

    def __init__(
        self,
        name: str,
        train_concepts: Sequence[Concept],
        test_concepts: Sequence[Concept],
        group_by_concept: Mapping[str, str],
        training_groups: Sequence[str],
        held_out_groups: Sequence[str],
        scenario: int,
        task_size: int,
    ):
        super().__init__(name=name, train_concepts=train_concepts, test_concepts=test_concepts)
        self._group_by_concept = dict(group_by_concept)
        self._training_groups = list(training_groups)
        self._held_out_groups = list(held_out_groups)
        self._scenario = scenario
        self._task_size = task_size

    def group_by_concept(self) -> Dict[str, str]:
        return dict(self._group_by_concept)

    def first_seen_step(self) -> Dict[str, int]:
        if self._held_out_groups:
            raise ValueError(
                f"Dataset {self.name()!r} holds out {self._held_out_groups} for zero-shot evaluation, so "
                "their classes never enter training and have no first seen step. Schedule-aware metrics "
                "describe trained concepts; read the dataset with zero_shot=False to use them."
            )
        step_by_group = {group: step for step, group in enumerate(self._training_groups)}
        return {concept: step_by_group[group] for concept, group in self._group_by_concept.items()}

    def held_out_groups(self) -> List[str]:
        return list(self._held_out_groups)

    def additional_info(self) -> Dict[str, Any]:
        return {
            **super().additional_info(),
            "scenario": self._scenario,
            "task_size": self._task_size,
            "held_out_groups": self._held_out_groups,
        }


class ContinualMegaBenchmarkReader:
    def __init__(
        self,
        data_root: Union[str, Path],
        meta_dir: Union[str, Path],
        scenario: int,
        task_size: int,
        zero_shot: bool = False,
        train_samples: str = "all",
        zero_shot_dirs: Mapping[str, str] = CONTINUAL_MEGA_ZERO_SHOT_DIRS,
    ):
        if scenario not in CONTINUAL_MEGA_SCENARIOS:
            raise ValueError(f"scenario must be one of {CONTINUAL_MEGA_SCENARIOS}, got {scenario}")
        if task_size not in CONTINUAL_MEGA_TASK_SIZES:
            raise ValueError(f"task_size must be one of {CONTINUAL_MEGA_TASK_SIZES}, got {task_size}")
        if train_samples not in CONTINUAL_MEGA_TRAIN_SAMPLES:
            raise ValueError(f"train_samples must be one of {CONTINUAL_MEGA_TRAIN_SAMPLES}, got {train_samples!r}")
        if zero_shot and scenario == 1:
            raise ValueError("Scenario 1 trains on MVTec-AD and VisA, so it has no held-out zero-shot datasets")

        self.data_root = Path(data_root).expanduser().resolve()
        self.meta_dir = Path(meta_dir).expanduser().resolve()
        self.scenario = scenario
        self.task_size = task_size
        self.zero_shot = zero_shot
        self.train_samples = train_samples
        self.zero_shot_dirs = dict(zero_shot_dirs)

    def index_groups(self) -> List[ConceptGroup]:
        groups = self._stream_groups()
        if self.zero_shot:
            groups.extend(self._held_out_groups())
        return groups

    def read_dataset(
        self,
        data_mode: str = "numpy",
        resize_to: Optional[Tuple[int, int]] = CONTINUAL_MEGA_IMAGE_SIZE,
        color_mode: str = "rgb",
        interpolation: str = CONTINUAL_MEGA_INTERPOLATION,
    ) -> ContinualMegaDataset:
        options = ImageLoadOptions(
            data_mode=data_mode,
            resize_to=resize_to,
            color_mode=color_mode,
            interpolation=interpolation,
            apply_exif_transpose=True,
        )
        groups = self.index_groups()

        return ContinualMegaDataset(
            name=f"Continual-MEGA-scenario{self.scenario}-{self.task_size}classes",
            train_concepts=LazyVisionConceptList(
                samples_by_concept=[(group.name, group.train_samples) for group in groups if not group.held_out],
                options=options,
                with_labels=True,
            ),
            test_concepts=LazyVisionConceptList(
                samples_by_concept=[concept for group in groups for concept in group.test_concepts],
                options=options,
                with_labels=True,
            ),
            group_by_concept={concept_name: group.name for group in groups for concept_name, _ in group.test_concepts},
            training_groups=[group.name for group in groups if not group.held_out],
            held_out_groups=[group.name for group in groups if group.held_out],
            scenario=self.scenario,
            task_size=self.task_size,
        )

    def _stream_groups(self) -> List[ConceptGroup]:
        base_meta = self._read_meta(f"scenario{self.scenario}_base.json")
        task_meta = self._read_meta(f"scenario{self.scenario}_{self.task_size}classes_tasks.json")
        task_names = sorted(task_meta, key=lambda name: int(name.split("_")[1]))

        groups = [self._stream_group(CONTINUAL_MEGA_BASE_GROUP, base_meta)]
        groups.extend(self._stream_group(name, task_meta[name]) for name in task_names)
        return groups

    def _stream_group(self, name: str, meta: Mapping[str, Any]) -> ConceptGroup:
        train: List[VisionSample] = []
        for category, entries in meta["train"].items():
            train.extend(_samples_from_meta(entries, "train", self.data_root, category))
        if self.train_samples == "normal":
            train = [sample for sample in train if sample.image_label == 0]

        return ConceptGroup(
            name=name,
            train_samples=train,
            test_concepts=[
                (category, _samples_from_meta(entries, "test", self.data_root, category))
                for category, entries in meta["test"].items()
            ],
        )

    def _held_out_groups(self) -> List[ConceptGroup]:
        mvtec_meta = self._read_meta("meta_mvtec.json")
        mvtec_root = self.data_root / self.zero_shot_dirs["mvtec"]
        mvtec_concepts = [
            (f"mvtec_{category}", _samples_from_meta(entries, "test", mvtec_root, category))
            for category, entries in mvtec_meta["test"].items()
        ]

        visa_reader = VisABenchmarkReader(root=self.data_root / self.zero_shot_dirs["visa"])
        visa_concepts: Dict[str, List[VisionSample]] = {}
        for sample in visa_reader.index_samples():
            if sample.split == "test":
                visa_concepts.setdefault(f"visa_{sample.category}", []).append(sample)

        return [
            ConceptGroup(name="zeroshot_mvtec", train_samples=[], test_concepts=mvtec_concepts, held_out=True),
            ConceptGroup(
                name="zeroshot_visa", train_samples=[], test_concepts=list(visa_concepts.items()), held_out=True
            ),
        ]

    def _read_meta(self, filename: str) -> Dict[str, Any]:
        path = self.meta_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Continual-MEGA meta file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


def _samples_from_meta(
    entries: Sequence[Mapping[str, Any]],
    split: str,
    root: Path,
    category: str,
) -> List[VisionSample]:
    return [
        VisionSample(
            category=category,
            split=split,
            image_path=root / entry["img_path"],
            image_label=int(entry["anomaly"]),
            mask_path=root / entry["mask_path"] if entry["mask_path"] else None,
            defect_type=entry.get("specie_name") or None,
        )
        for entry in entries
    ]
