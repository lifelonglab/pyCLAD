"""Read COMMAND's RGB/flow UCF-Crime archive as complete video bags."""

import ast
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Sequence, Tuple, Union
from zipfile import ZipFile

import numpy as np
from filelock import FileLock
from huggingface_hub import hf_hub_download

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.video.data.sample import VideoBag, VideoWindow
from pyclad.video.data.video_bag_concept import VideoBagConcept

# COMMAND Section IV-C groups the anomaly classes into 4/4/5 tasks.
COMMAND_UCF_CRIME_PAPER_TASKS = (
    ("Abuse", "Arrest", "Arson", "Assault"),
    ("Burglary", "Explosion", "Fighting", "RoadAccidents"),
    ("Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"),
)
COMMAND_UCF_CRIME_CONCEPT_ORDER = tuple(category for task in COMMAND_UCF_CRIME_PAPER_TASKS for category in task)


@dataclass(frozen=True)
class CommandUcfCrimeRecord:
    """One video entry from COMMAND's UCF-Crime split files."""

    relative_path: str
    weak_label: int
    anomaly_class: Optional[str]
    frame_count: Optional[int] = None
    anomaly_intervals: Tuple[Tuple[int, int], ...] = ()

    @property
    def video_id(self) -> str:
        return self.relative_path


class CommandUcfCrimeDataset(ConceptsDataset):
    """Download/cache paired RGB/flow features or read a local COMMAND archive.

    The 4/4/5 stream retains repeated manifest rows as distinct training bags.
    """

    _hf_repo = "nmcnamee24/ucf-crime-rgb-flow-features"

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        cache_dir: Optional[Union[str, Path]] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        max_videos_per_class: Optional[int] = None,
    ):
        """
        :param root: Existing archive directory. If omitted, download from Hugging Face.
            An explicit local path never triggers a download.
        :param cache_dir: Hugging Face cache directory; None uses its default cache.
        :param revision: Optional Hub commit, tag, or branch; None uses the latest main.
        :param local_files_only: Use only a previously downloaded Hub snapshot.
        :param max_videos_per_class: Optional positive training limit per class and label;
            the complete test split is retained.
        """
        if root is None:
            root = self._download_root(
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
            )
        self.root = Path(root).expanduser().resolve()
        self._validate_layout(self.root)
        self._normal_train = self._read_train_records("train_normal.txt", weak_label=0)
        self._anomaly_train = self._read_train_records("train_anomaly.txt", weak_label=1)
        self._test_records = (*self._read_normal_test_records(), *self._read_anomaly_test_records())
        if len(self._normal_train) != len(self._anomaly_train):
            raise ValueError("normal and anomaly training manifest counts must match")
        unknown = {r.anomaly_class for r in self._anomaly_train} - set(COMMAND_UCF_CRIME_CONCEPT_ORDER)
        if unknown:
            raise ValueError(f"unknown anomaly classes: {sorted(unknown)}")
        train_ids = {r.video_id for r in (*self._normal_train, *self._anomaly_train)}
        test_ids = [r.video_id for r in self._test_records]
        if train_ids.intersection(test_ids):
            raise ValueError("training and test manifests overlap")
        if len(set(test_ids)) != len(test_ids):
            raise ValueError("test manifest contains duplicate videos")
        if any(r.frame_count < 32 for r in self._test_records):
            raise ValueError("test videos must have at least 32 frames")
        self._record_ids = {
            id(r): f"train:{i}:{r.video_id}" for i, r in enumerate((*self._normal_train, *self._anomaly_train))
        }
        dataset = self._read_dataset(max_videos_per_class=max_videos_per_class)
        super().__init__(
            name=dataset.name(), train_concepts=dataset.train_concepts(), test_concepts=dataset.test_concepts()
        )

    @classmethod
    def _download_root(
        cls,
        *,
        cache_dir: Optional[Union[str, Path]],
        revision: Optional[str],
        local_files_only: bool,
    ) -> Path:
        """Download one ZIP and atomically cache its extracted, revision-specific layout."""
        archive = Path(
            hf_hub_download(
                repo_id=cls._hf_repo,
                filename="ucf-crime-rgb-flow.zip",
                repo_type="dataset",
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
        # Keep the snapshot path rather than resolving its symlink into the blob store.
        root = archive.with_suffix("")
        with FileLock(str(root) + ".extract.lock"):
            if not root.is_dir():
                with TemporaryDirectory(prefix=f".{root.name}-", dir=archive.parent) as directory:
                    temporary = Path(directory).resolve()
                    with ZipFile(archive) as zipped:
                        for member in zipped.infolist():
                            if not (temporary / member.filename).resolve().is_relative_to(temporary):
                                raise ValueError(f"unsafe archive member: {member.filename!r}")
                        zipped.extractall(temporary)
                    cls._validate_layout(temporary)
                    temporary.rename(root)
        return root

    def read_dataset(self, *, max_videos_per_class: Optional[int] = None) -> ConceptsDataset:
        """Compatibility wrapper; the constructor already exposes the concepts."""
        if max_videos_per_class is None:
            return self
        return self._read_dataset(max_videos_per_class=max_videos_per_class)

    def _read_dataset(self, *, max_videos_per_class: Optional[int] = None) -> ConceptsDataset:
        """Materialize T1/T2/T3 with disjoint normal shards and a fixed test split.

        An optional positive per-class limit supports small smoke runs. It is
        applied after normal allocation, so truncation cannot change task identity.
        """
        if max_videos_per_class is not None and max_videos_per_class <= 0:
            raise ValueError("max_videos_per_class must be positive")
        by_class = {}
        offset = 0
        for category in COMMAND_UCF_CRIME_CONCEPT_ORDER:
            anomalies = [r for r in self._anomaly_train if r.anomaly_class == category]
            normals = self._normal_train[offset : offset + len(anomalies)]
            offset += len(anomalies)
            by_class[category] = (*anomalies[:max_videos_per_class], *normals[:max_videos_per_class])
        train = []
        for index, categories in enumerate(COMMAND_UCF_CRIME_PAPER_TASKS, 1):
            records = [r for category in categories for r in by_class[category]]
            if not records:
                raise ValueError(f"no training videos for task T{index}")
            train.append(self._concept(f"T{index}", records, training=True))
        if not self._test_records:
            raise ValueError("test manifest is empty")
        test = self._concept("test", self._test_records, training=False)
        return ConceptsDataset("COMMAND-UCF-Crime", train, [test])

    def _concept(self, name: str, records: Sequence[CommandUcfCrimeRecord], *, training: bool) -> VideoBagConcept:
        bags = []
        frame_labels = {}
        for record in records:
            arrays = [
                np.load(self.root / stream / f"{record.relative_path}.npy", allow_pickle=False)
                for stream in ("all_rgbs", "all_flows")
            ]
            if any(a.shape != (32, 1024) or not np.isfinite(a).all() for a in arrays):
                raise ValueError(f"expected finite (32, 1024) RGB/flow arrays for {record.video_id}")
            windows = tuple(VideoWindow(record.video_id, start, stop) for start, stop in self._frame_ranges(record, 32))
            bags.append(
                VideoBag(
                    bag_id=self._record_ids[id(record)] if training else record.video_id,
                    task_id=name,
                    features=np.concatenate(arrays, axis=1),
                    weak_label=record.weak_label,
                    windows=windows,
                )
            )
            if not training:
                frame_labels[record.video_id] = self._frame_labels_for_record(record)
        return VideoBagConcept(
            name=name,
            data=np.asarray(bags, dtype=object),
            labels=np.asarray([b.weak_label for b in bags]),
            frame_labels=frame_labels,
        )

    @staticmethod
    def _frame_ranges(
        record: CommandUcfCrimeRecord,
        window_count: int,
    ) -> Tuple[Tuple[int, int], ...]:
        """Keep COMMAND's rounded 16-frame clip grid and score the residual tail.

        The last window extends over any trailing incomplete clip, preserving
        complete frame coverage. Videos shorter than ``window_count * 16``
        frames use rounded raw-frame bins: the clip grid cannot provide one
        nonempty window per feature row without overlaps for these videos.
        """
        if record.frame_count is None:
            return tuple((index, index) for index in range(window_count))
        clip_count = record.frame_count // 16
        if clip_count >= window_count:
            edges = np.rint(np.linspace(0, clip_count, window_count + 1)).astype(np.int64) * 16
            edges[-1] = record.frame_count
        else:
            edges = np.rint(np.linspace(0, record.frame_count, window_count + 1)).astype(np.int64)
        return tuple(
            (int(edges[index]), max(int(edges[index]), int(edges[index + 1]) - 1)) for index in range(window_count)
        )

    @staticmethod
    def _frame_labels_for_record(record: CommandUcfCrimeRecord) -> np.ndarray:
        if record.frame_count is None:
            raise ValueError(f"frame count is missing for {record.video_id}")
        labels = np.zeros(record.frame_count, dtype=np.int64)
        for start, stop in record.anomaly_intervals:
            labels[max(0, start) : min(stop, record.frame_count)] = 1
        return labels

    def _read_train_records(
        self,
        filename: str,
        *,
        weak_label: int,
    ) -> Tuple[CommandUcfCrimeRecord, ...]:
        records = []
        with (self.root / filename).open(encoding="utf-8") as stream:
            for raw_line in stream:
                relative_path = raw_line.strip()
                if not relative_path:
                    continue
                anomaly_class = relative_path.split("/", 1)[0] if weak_label else None
                records.append(
                    CommandUcfCrimeRecord(
                        relative_path=relative_path,
                        weak_label=weak_label,
                        anomaly_class=anomaly_class,
                    )
                )
        return tuple(records)

    def _read_normal_test_records(self) -> Tuple[CommandUcfCrimeRecord, ...]:
        records = []
        with (self.root / "test_normalv2.txt").open(encoding="utf-8") as stream:
            for raw_line in stream:
                fields = raw_line.split()
                if not fields:
                    continue
                if len(fields) != 3:
                    raise ValueError(f"invalid COMMAND normal test row: {raw_line.rstrip()!r}")
                records.append(
                    CommandUcfCrimeRecord(
                        relative_path=fields[0],
                        weak_label=0,
                        anomaly_class=None,
                        frame_count=int(fields[1]),
                    )
                )
        return tuple(records)

    def _read_anomaly_test_records(self) -> Tuple[CommandUcfCrimeRecord, ...]:
        records = []
        with (self.root / "test_anomalyv2.txt").open(encoding="utf-8") as stream:
            for raw_line in stream:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                fields = raw_line.split("|")
                if len(fields) != 3:
                    raise ValueError(f"invalid COMMAND anomaly test row: {raw_line!r}")
                boundaries = tuple(int(value) for value in ast.literal_eval(fields[2]))
                if len(boundaries) % 2:
                    raise ValueError(f"anomaly boundaries must contain start/stop pairs: {raw_line!r}")
                if any(start < 1 or stop < start for start, stop in zip(boundaries[::2], boundaries[1::2])):
                    raise ValueError(f"anomaly intervals must use positive, ordered frame coordinates: {raw_line!r}")
                intervals = tuple(
                    (boundaries[index] - 1, boundaries[index + 1]) for index in range(0, len(boundaries), 2)
                )
                anomaly_class = fields[0].split("/", 1)[0]
                records.append(
                    CommandUcfCrimeRecord(
                        relative_path=fields[0],
                        weak_label=1,
                        anomaly_class=anomaly_class,
                        frame_count=int(fields[1]),
                        anomaly_intervals=intervals,
                    )
                )
        return tuple(records)

    @staticmethod
    def _validate_layout(root: Path) -> None:
        """Reject incomplete local or extracted feature layouts."""
        required = (
            root / "all_rgbs",
            root / "all_flows",
            root / "train_normal.txt",
            root / "train_anomaly.txt",
            root / "test_normalv2.txt",
            root / "test_anomalyv2.txt",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"COMMAND UCF-Crime layout is incomplete; missing: {missing}")
