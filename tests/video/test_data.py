"""Archive annotation and task-allocation regressions."""

from zipfile import ZipFile

import numpy as np
import pytest

from pyclad.video.data.command_ucf_crime import CommandUcfCrimeDataset
from pyclad.video.metrics.frame_score_utils import window_scores_to_frame_scores


def write_feature_zip(root, destination):
    """Pack the paired test features and original split lists into one Hub artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [p for stream in ("all_rgbs", "all_flows") for p in (root / stream).rglob("*.npy")]
    files += [
        root / name for name in ("train_normal.txt", "train_anomaly.txt", "test_normalv2.txt", "test_anomalyv2.txt")
    ]
    with ZipFile(destination, "w") as zipped:
        for path in files:
            zipped.write(path, path.relative_to(root).as_posix())
    return destination


@pytest.fixture(autouse=True)
def reject_per_feature_download(monkeypatch):
    """Unit tests must never fetch the individual remote feature files."""
    from pyclad.video.data import command_ucf_crime

    def unexpected_snapshot(**kwargs):
        pytest.fail("the loader must download a single archive rather than a per-feature snapshot")

    monkeypatch.setattr(command_ucf_crime, "snapshot_download", unexpected_snapshot, raising=False)


@pytest.fixture
def archive(tmp_path):
    anomalies = [f"{category}/train{index}.mp4" for category in ("Abuse", "Burglary", "Robbery") for index in range(2)]
    normals = [f"Normal_Videos_event/train{index}.mp4" for index in range(6)]
    test = ["Normal_Videos_event/test.mp4", "Abuse/test.mp4"]
    for stream in ("all_rgbs", "all_flows"):
        for name in anomalies + normals + test:
            path = tmp_path / stream / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.zeros((32, 1024), dtype=np.float32))
    (tmp_path / "train_normal.txt").write_text("\n".join(normals) + "\n")
    (tmp_path / "train_anomaly.txt").write_text("\n".join(anomalies) + "\n")
    (tmp_path / "test_normalv2.txt").write_text(f"{test[0]} 67 -1\n")
    (tmp_path / "test_anomalyv2.txt").write_text(f"{test[1]}|67|[1, 1, 33, 48, 65, 68]\n")
    return tmp_path


def test_annotations_and_uneven_windows_align_on_frames(archive):
    """Short videos retain 32 disjoint raw-frame bins and unchanged annotations."""
    concept = CommandUcfCrimeDataset(archive).read_dataset().test_concepts()[0]
    bag = concept.data[1]
    expected_labels = np.zeros(67, dtype=np.int64)
    expected_labels[0] = 1
    expected_labels[32:48] = 1
    expected_labels[64:] = 1
    np.testing.assert_array_equal(concept.frame_labels[bag.bag_id], expected_labels)

    # The release contains intervals ending one frame beyond the video; retain clipping.
    scores = np.arange(32, dtype=np.float64)
    curves = window_scores_to_frame_scores(bag.windows, scores, {bag.bag_id: 67})
    widths = [2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2]
    np.testing.assert_array_equal(curves[bag.bag_id], np.repeat(scores, widths))
    assert len(bag.windows) == len(bag.features) == 32
    assert all(left.end_frame + 1 == right.start_frame for left, right in zip(bag.windows, bag.windows[1:]))
    assert bag.windows[0].start_frame == 0
    assert bag.windows[-1].end_frame == 66


@pytest.mark.parametrize("frame_count, tail_frames", [(768, 0), (777, 9)])
def test_clip_grid_preserves_window_scores_and_covers_residual_tail(archive, frame_count, tail_frames):
    """The release's 777-frame case uses clip boundaries, including its nine-frame tail."""
    (archive / "test_anomalyv2.txt").write_text(f"Abuse/test.mp4|{frame_count}|[105, 180]\n")
    concept = CommandUcfCrimeDataset(archive).test_concepts()[0]
    bag = concept.data[1]

    assert [window.start_frame for window in bag.windows[:4]] == [0, 32, 48, 64]
    assert len(bag.windows) == len(bag.features) == 32
    assert all(left.end_frame + 1 == right.start_frame for left, right in zip(bag.windows, bag.windows[1:]))
    assert bag.windows[-1].end_frame == frame_count - 1

    # Hand-derived widths from 48 clips over 32 bins, with round-to-even ties.
    widths = [32, 16, 16, 32] * 8
    widths[-1] += tail_frames
    scores = np.arange(32, dtype=np.float64)
    curves = window_scores_to_frame_scores(bag.windows, scores, {bag.bag_id: frame_count})
    np.testing.assert_array_equal(curves[bag.bag_id], np.repeat(scores, widths))
    expected_labels = np.zeros(frame_count, dtype=np.int64)
    expected_labels[104:180] = 1
    np.testing.assert_array_equal(concept.frame_labels[bag.bag_id], expected_labels)


@pytest.mark.parametrize("frame_count, last_width", [(512, 16), (527, 31)])
def test_one_clip_per_window_starts_at_512_frames(archive, frame_count, last_width):
    """At the clip-grid threshold, all 32 scores own nonempty disjoint windows."""
    (archive / "test_normalv2.txt").write_text(f"Normal_Videos_event/test.mp4 {frame_count} -1\n")
    bag = CommandUcfCrimeDataset(archive).test_concepts()[0].data[0]
    scores = np.arange(32, dtype=np.float64)
    curves = window_scores_to_frame_scores(bag.windows, scores, {bag.bag_id: frame_count})
    np.testing.assert_array_equal(curves[bag.bag_id], np.repeat(scores, [16] * 31 + [last_width]))


def test_limit_preserves_normal_task_allocation_and_test_split(archive):
    reader = CommandUcfCrimeDataset(archive)
    full = reader.read_dataset()
    limited = reader.read_dataset(max_videos_per_class=1)
    normal_ids = []
    for complete, small in zip(full.train_concepts(), limited.train_concepts()):
        assert complete.name == small.name
        assert len(complete.data) == 4
        assert len(small.data) == 2
        full_normals = [bag.bag_id for bag in complete.data if bag.weak_label == 0]
        small_normals = [bag.bag_id for bag in small.data if bag.weak_label == 0]
        assert small_normals == full_normals[:1]
        normal_ids.extend(full_normals)
    assert len(set(normal_ids)) == 6
    assert [bag.bag_id for bag in full.test_concepts()[0].data] == [
        bag.bag_id for bag in limited.test_concepts()[0].data
    ]


@pytest.mark.parametrize("boundaries", ["[-1, -1]", "[0, 4]", "[8, 7]"])
def test_invalid_annotations_cannot_silently_change_frame_labels(archive, boundaries):
    (archive / "test_anomalyv2.txt").write_text(f"Abuse/test.mp4|67|{boundaries}\n")
    with pytest.raises(ValueError, match="positive, ordered frame coordinates"):
        CommandUcfCrimeDataset(archive)


@pytest.mark.parametrize("custom_cache", [False, True])
def test_hub_download_loads_paired_features_and_original_tasks(archive, monkeypatch, custom_cache):
    """The downloaded snapshot must feed the same RGB-first bag reader as local data."""
    from pyclad.video.data import command_ucf_crime

    for stream, value in (("all_rgbs", 1.0), ("all_flows", 2.0)):
        np.save(archive / stream / "Abuse/train0.mp4.npy", np.full((32, 1024), value, dtype=np.float32))
    options = dict(cache_dir=archive / "cache", revision="test-revision", local_files_only=True) if custom_cache else {}
    feature_zip = write_feature_zip(archive, archive / "hub-snapshot" / "ucf-crime-rgb-flow.zip")

    def download(*, repo_id, filename, repo_type, cache_dir, revision, local_files_only):
        assert repo_id == "nmcnamee24/ucf-crime-rgb-flow-features"
        assert repo_type == "dataset"
        assert cache_dir == options.get("cache_dir")
        assert revision == options.get("revision")
        assert local_files_only == options.get("local_files_only", False)
        assert filename == "ucf-crime-rgb-flow.zip"
        return str(feature_zip)

    monkeypatch.setattr(command_ucf_crime, "hf_hub_download", download, raising=False)
    dataset = CommandUcfCrimeDataset(**options)
    assert [c.name for c in dataset.train_concepts()] == ["T1", "T2", "T3"]
    assert [len(c.data) for c in dataset.train_concepts()] == [4, 4, 4]
    assert len(dataset.test_concepts()[0].data) == 2
    bag = dataset.train_concepts()[0].data[0]
    assert bag.features.shape == (32, 2048)
    np.testing.assert_array_equal(bag.features[:, :1024], 1.0)
    np.testing.assert_array_equal(bag.features[:, 1024:], 2.0)
    assert dataset.root != archive


def test_extracted_features_are_reused_without_extracting_again(archive, monkeypatch):
    from pyclad.video.data import command_ucf_crime

    feature_zip = write_feature_zip(archive, archive / "snapshot" / "ucf-crime-rgb-flow.zip")
    monkeypatch.setattr(command_ucf_crime, "hf_hub_download", lambda **kwargs: str(feature_zip), raising=False)
    first = CommandUcfCrimeDataset()
    # The extracted cache should remain usable without reopening the ZIP.
    feature_zip.write_bytes(b"not a ZIP")
    cached = CommandUcfCrimeDataset(local_files_only=True)
    assert cached.root == first.root
    np.testing.assert_array_equal(
        cached.train_concepts()[0].data[0].features, first.train_concepts()[0].data[0].features
    )


def test_archive_revisions_use_separate_extracted_caches(archive, monkeypatch):
    from pyclad.video.data import command_ucf_crime

    first_zip = write_feature_zip(archive, archive / "snapshots/v1/ucf-crime-rgb-flow.zip")
    np.save(archive / "all_rgbs/Abuse/train0.mp4.npy", np.ones((32, 1024), dtype=np.float32))
    second_zip = write_feature_zip(archive, archive / "snapshots/v2/ucf-crime-rgb-flow.zip")
    monkeypatch.setattr(
        command_ucf_crime,
        "hf_hub_download",
        lambda **kwargs: str(first_zip if kwargs["revision"] == "v1" else second_zip),
        raising=False,
    )
    first = CommandUcfCrimeDataset(revision="v1")
    second = CommandUcfCrimeDataset(revision="v2")
    assert first.root != second.root
    np.testing.assert_array_equal(first.train_concepts()[0].data[0].features[:, :1024], 0.0)
    np.testing.assert_array_equal(second.train_concepts()[0].data[0].features[:, :1024], 1.0)


def test_simultaneous_loaders_share_one_complete_extraction(archive, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from pyclad.video.data import command_ucf_crime

    feature_zip = write_feature_zip(archive, archive / "snapshot/ucf-crime-rgb-flow.zip")
    monkeypatch.setattr(command_ucf_crime, "hf_hub_download", lambda **kwargs: str(feature_zip))
    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(lambda _: CommandUcfCrimeDataset(), range(2)))
    assert first.root == second.root
    assert len(list(first.root.rglob("*.npy"))) == 28
    np.testing.assert_array_equal(
        first.train_concepts()[0].data[0].features, second.train_concepts()[0].data[0].features
    )


@pytest.mark.parametrize("fault", ["unsafe_path", "incomplete_layout"])
def test_invalid_archive_is_not_published_as_a_complete_cache(archive, monkeypatch, fault):
    from pyclad.video.data import command_ucf_crime

    feature_zip = archive / "snapshot/ucf-crime-rgb-flow.zip"
    feature_zip.parent.mkdir()
    with ZipFile(feature_zip, "w") as zipped:
        zipped.writestr("../escaped.txt" if fault == "unsafe_path" else "train_normal.txt", "invalid")
    monkeypatch.setattr(command_ucf_crime, "hf_hub_download", lambda **kwargs: str(feature_zip), raising=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        CommandUcfCrimeDataset()
    assert not feature_zip.with_suffix("").exists()
    assert not (feature_zip.parent / "escaped.txt").exists()
    write_feature_zip(archive, feature_zip)
    dataset = CommandUcfCrimeDataset()
    assert [len(c.data) for c in dataset.train_concepts()] == [4, 4, 4]


def test_constructor_returns_ready_concepts_like_core_loaders(archive):
    """A freshly constructed loader exposes concepts and core output metadata."""
    dataset = CommandUcfCrimeDataset(archive)
    assert [c.name for c in dataset.train_concepts()] == ["T1", "T2", "T3"]
    assert [len(c.data) for c in dataset.train_concepts()] == [4, 4, 4]
    assert len(dataset.test_concepts()[0].data) == 2
    assert dataset.info() == {"dataset": {"name": "COMMAND-UCF-Crime", "train_concepts_no": 3, "test_concepts_no": 1}}
    assert dataset.read_dataset() is dataset


def test_constructor_can_limit_training_without_changing_test_split(archive):
    """Small runs use the same dataset contract and retain all test videos."""
    dataset = CommandUcfCrimeDataset(archive, max_videos_per_class=1)
    assert [len(c.data) for c in dataset.train_concepts()] == [2, 2, 2]
    assert len(dataset.test_concepts()[0].data) == 2


def test_explicit_local_root_never_downloads(archive, monkeypatch):
    """Local archives, including incomplete ones, must never fall back to the Hub."""
    from pyclad.video.data import command_ucf_crime

    def unexpected_download(**kwargs):
        pytest.fail("an explicit local root must not contact Hugging Face")

    monkeypatch.setattr(command_ucf_crime, "hf_hub_download", unexpected_download, raising=False)
    dataset = CommandUcfCrimeDataset(archive, cache_dir=archive / "unused").read_dataset()
    assert [len(c.data) for c in dataset.train_concepts()] == [4, 4, 4]
    with pytest.raises(FileNotFoundError, match="layout is incomplete"):
        CommandUcfCrimeDataset(archive / "missing")


def test_hub_download_failure_preserves_actionable_error(monkeypatch):
    """Do not turn a failed download into a misleading local-layout error."""
    from pyclad.video.data import command_ucf_crime

    def failed_download(**kwargs):
        raise ConnectionError("Cannot reach Hugging Face; retry or use a cached snapshot")

    monkeypatch.setattr(command_ucf_crime, "hf_hub_download", failed_download, raising=False)
    with pytest.raises(ConnectionError, match="Cannot reach Hugging Face"):
        CommandUcfCrimeDataset()


@pytest.mark.longrun
def test_downloaded_release_and_offline_cache_have_all_three_tasks(tmp_path):
    """Exercise the published paired release through the public dataset loader."""
    options = dict(cache_dir=tmp_path, revision="96d72a51d0f62b0b08f9d15d4280ced1517dbe1a")
    reader = CommandUcfCrimeDataset(**options)
    dataset = reader.read_dataset()
    assert [c.name for c in dataset.train_concepts()] == ["T1", "T2", "T3"]
    assert [len(c.data) for c in dataset.train_concepts()] == [362, 576, 682]
    assert len(dataset.test_concepts()[0].data) == 290
    for concept in dataset.train_concepts() + dataset.test_concepts():
        assert all(bag.features.shape == (32, 2048) and np.isfinite(bag.features).all() for bag in concept.data)
    cached = CommandUcfCrimeDataset(**options, local_files_only=True)
    assert cached.root == reader.root
    limited = cached.read_dataset(max_videos_per_class=1)
    assert [len(c.data) for c in limited.train_concepts()] == [8, 8, 10]
    np.testing.assert_array_equal(
        limited.train_concepts()[0].data[0].features, dataset.train_concepts()[0].data[0].features
    )
