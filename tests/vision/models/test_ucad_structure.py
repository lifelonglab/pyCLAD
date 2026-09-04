import numpy as np
import pytest
from PIL import Image

from pyclad.vision.models.ucad.config import UCADConfig
from pyclad.vision.models.ucad.structure import (
    NoStructureMaskProvider,
    PrecomputedStructureMaskProvider,
    build_structure_mask_provider,
    category_of,
)


def _write_label_map(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(path)


def test_category_of_strips_a_multidataset_prefix():
    assert category_of("mvtec__bottle") == "bottle"
    assert category_of("bottle") == "bottle"


def test_no_provider_returns_none():
    provider = NoStructureMaskProvider()

    assert provider.masks_for("bottle", np.zeros((2, 4, 4, 3), dtype=np.uint8)) is None


def test_precomputed_provider_loads_in_sorted_order(tmp_path):
    directory = tmp_path / "bottle" / "train" / "good"
    _write_label_map(directory / "001.png", [[2, 2], [2, 2]])
    _write_label_map(directory / "000.png", [[1, 1], [1, 1]])

    provider = PrecomputedStructureMaskProvider(tmp_path)
    masks = provider.masks_for("bottle", np.zeros((2, 2, 2, 3), dtype=np.uint8))

    assert masks.shape == (2, 2, 2)
    assert masks.dtype == np.int64
    assert masks[0].max() == 1  # 000.png sorts first
    assert masks[1].max() == 2


def test_precomputed_provider_resolves_a_multidataset_concept_id(tmp_path):
    _write_label_map(tmp_path / "bottle" / "train" / "good" / "000.png", [[3, 3], [3, 3]])

    provider = PrecomputedStructureMaskProvider(tmp_path)
    masks = provider.masks_for("mvtec__bottle", np.zeros((1, 2, 2, 3), dtype=np.uint8))

    assert masks[0].max() == 3


def test_precomputed_provider_resizes_to_the_image_grid(tmp_path):
    _write_label_map(tmp_path / "bottle" / "000.png", [[1, 1], [1, 1]])

    provider = PrecomputedStructureMaskProvider(tmp_path)
    masks = provider.masks_for("bottle", np.zeros((1, 8, 6, 3), dtype=np.uint8))

    assert masks.shape == (1, 8, 6)


def test_precomputed_provider_reads_channel_0_not_luma_for_rgb_label_maps(tmp_path):
    # Pins the reference's cv2.imread(...)[:, :, 0] semantics: ids must come from channel 0
    # untouched, not from an RGB->luma blend with the other two channels.
    directory = tmp_path / "bottle"
    directory.mkdir(parents=True, exist_ok=True)
    ids = np.array([[10, 200], [50, 1]], dtype=np.uint8)
    other = np.full_like(ids, 128)  # far enough from channel 0 that luma blending is obvious
    rgb = np.stack([ids, other, other], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(directory / "000.png")

    provider = PrecomputedStructureMaskProvider(tmp_path)
    masks = provider.masks_for("bottle", np.zeros((1, 2, 2, 3), dtype=np.uint8))

    assert np.array_equal(masks[0], ids.astype(np.int64))


def test_precomputed_provider_nearest_interpolation_preserves_source_ids(tmp_path):
    # Nearest neighbor must not blend adjacent distinct ids into interpolated in-between
    # values the way bilinear does when upsampling.
    _write_label_map(tmp_path / "bottle" / "000.png", [[1, 1], [200, 200]])

    nearest_provider = PrecomputedStructureMaskProvider(tmp_path, interpolation="nearest")
    nearest_masks = nearest_provider.masks_for("bottle", np.zeros((1, 8, 2, 3), dtype=np.uint8))

    assert nearest_masks.shape == (1, 8, 2)
    assert set(np.unique(nearest_masks[0])) == {1, 200}

    # Sanity check that this scenario actually exercises interpolation: the default
    # bilinear path introduces values absent from the source, unlike nearest above.
    bilinear_provider = PrecomputedStructureMaskProvider(tmp_path)
    bilinear_masks = bilinear_provider.masks_for("bottle", np.zeros((1, 8, 2, 3), dtype=np.uint8))

    assert not set(np.unique(bilinear_masks[0])) <= {1, 200}


def test_precomputed_provider_rejects_a_count_mismatch(tmp_path):
    _write_label_map(tmp_path / "bottle" / "000.png", [[1, 1], [1, 1]])

    provider = PrecomputedStructureMaskProvider(tmp_path)

    with pytest.raises(ValueError, match="1 structure mask"):
        provider.masks_for("bottle", np.zeros((3, 2, 2, 3), dtype=np.uint8))


def test_precomputed_provider_reports_the_directories_it_tried(tmp_path):
    provider = PrecomputedStructureMaskProvider(tmp_path)

    with pytest.raises(FileNotFoundError, match="Checked:"):
        provider.masks_for("bottle", np.zeros((1, 2, 2, 3), dtype=np.uint8))


def test_precomputed_provider_rejects_a_missing_concept_id(tmp_path):
    # Reachable by running UCADStrategy under ConceptIncrementalScenario (which it declares
    # support for) with structure_mode="precomputed": no concept id ever reaches fit(), so
    # `names` would stay empty and produce an uninformative "Checked: " with nothing after it.
    provider = PrecomputedStructureMaskProvider(tmp_path)

    with pytest.raises(ValueError, match="needs a concept id"):
        provider.masks_for(None, np.zeros((1, 2, 2, 3), dtype=np.uint8))


def test_precomputed_provider_rejects_an_unknown_interpolation_mode(tmp_path):
    with pytest.raises(ValueError, match="interpolation"):
        PrecomputedStructureMaskProvider(tmp_path, interpolation="cubic")


def test_builder_returns_the_no_op_provider_for_the_cpm_only_ablation():
    provider = build_structure_mask_provider(UCADConfig(structure_mode="none"), "cpu")

    assert isinstance(provider, NoStructureMaskProvider)


def test_builder_returns_the_precomputed_provider(tmp_path):
    config = UCADConfig(structure_mode="precomputed", structure_mask_root=str(tmp_path))

    assert isinstance(build_structure_mask_provider(config, "cpu"), PrecomputedStructureMaskProvider)
