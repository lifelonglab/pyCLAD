from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from pyclad.vision.strategies.replaycad.config import ReplayCADConfig
from pyclad.vision.strategies.replaycad.masks import (
    FullFrameMaskProvider,
    PrecomputedMaskProvider,
    SamMaskProvider,
    augment_mask,
    augment_training_pair,
    build_mask_provider,
    little_rotate_and_move,
    random_3directions_rotate,
    random_reset,
    random_rotate,
    select_stored_masks,
    visa_candle,
)


def _config(tmp_path: Path, **overrides) -> ReplayCADConfig:
    defaults = dict(mask_backend="full-frame")
    defaults.update(overrides)
    return ReplayCADConfig.for_benchmark("mvtec", artifact_dir=tmp_path, **defaults)


def test_full_frame_provider_returns_all_ones():
    images = np.zeros((2, 16, 16, 3), dtype=np.uint8)

    masks = FullFrameMaskProvider().masks_for("screw", images)

    assert masks.shape == (2, 16, 16)
    assert np.all(masks == 255)


def test_precomputed_provider_reads_sorted_files(tmp_path: Path):
    category_dir = tmp_path / "mvtec" / "screw"
    category_dir.mkdir(parents=True)
    for index, value in enumerate([0, 128, 255]):
        Image.fromarray(np.full((16, 16), value, dtype=np.uint8)).save(category_dir / f"{index:03d}.png")

    provider = PrecomputedMaskProvider(root=tmp_path, benchmark="mvtec")
    masks = provider.masks_for("mvtec__screw", np.zeros((3, 16, 16, 3), dtype=np.uint8))

    assert masks.shape == (3, 16, 16)
    assert masks[0].max() == 0 and masks[2].max() == 255
    # The 128-valued middle file must come back binarized, not passed through.
    assert set(np.unique(masks)).issubset({0, 255})
    assert masks[1].max() == 255


def test_precomputed_provider_rejects_a_count_mismatch(tmp_path: Path):
    category_dir = tmp_path / "mvtec" / "screw"
    category_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(category_dir / "000.png")

    provider = PrecomputedMaskProvider(root=tmp_path, benchmark="mvtec")

    with pytest.raises(ValueError, match="1 precomputed mask"):
        provider.masks_for("mvtec__screw", np.zeros((3, 16, 16, 3), dtype=np.uint8))


def test_mask_modes_route_listed_concepts_to_full_frame(tmp_path: Path):
    # carpet is a texture: SAM returns no object mask, so it is pinned to the full frame while
    # every other concept still goes to the configured backend.
    config = _config(
        tmp_path,
        mask_backend="precomputed",
        precomputed_mask_root=tmp_path / "sam",
        mask_modes={"carpet": "full-frame"},
    )
    provider = build_mask_provider(config, benchmark="mvtec")
    images = np.zeros((2, 16, 16, 3), dtype=np.uint8)

    routed = provider.masks_for("mvtec__carpet", images)

    assert np.all(routed == 255)
    with pytest.raises(ValueError, match="No precomputed masks"):
        provider.masks_for("mvtec__screw", images)


def test_sam_backend_is_selected_without_importing_segment_anything(tmp_path: Path):
    # Constructing the provider must not need the optional dependency; only masks_for() does.
    config = _config(tmp_path, mask_backend="sam", sam_checkpoint=tmp_path / "sam_vit_h.pth")

    provider = build_mask_provider(config, benchmark="mvtec")

    assert isinstance(provider, SamMaskProvider)


def test_augmentation_is_a_no_op_by_default(tmp_path: Path):
    config = _config(tmp_path)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255

    assert np.array_equal(augment_mask(mask, config, np.random.default_rng(0)), mask)


def test_paper_augmentation_rotates_and_shifts_deterministically(tmp_path: Path):
    config = _config(tmp_path, mask_augmentation="paper")
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    first = augment_mask(mask, config, np.random.default_rng(0))
    again = augment_mask(mask, config, np.random.default_rng(0))
    other = augment_mask(mask, config, np.random.default_rng(1))

    assert np.array_equal(first, again)  # same seed -> same augmentation
    assert not np.array_equal(first, other)
    assert not np.array_equal(first, mask)
    assert first.shape == mask.shape
    assert set(np.unique(first)).issubset({0, 255})  # stays a binary mask


def _two_blob_mask(size: int = 40) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[10:18, 10:18] = 255
    mask[size - 15 : size - 7, 5:9] = 255
    return mask


class _RecordingRng:
    """Wraps a real ``Generator``, logging the arguments and result of every ``integers``/
    ``uniform``/``choice`` call.

    Some of the mask transforms' RNG bounds are off-by-one mutations away from a range whose
    single missing value renders identically to a neighbour once drawn, rotated and binarized (for
    example ``random_rotate``'s top-of-range angle 360 looks exactly like angle 0). No amount of
    output-level comparison can distinguish those cases statistically without an impractical
    number of samples, so a handful of tests below pin the exact bound passed to the RNG instead.
    """

    def __init__(self, rng: np.random.Generator):
        self._rng = rng
        self.calls: list[tuple] = []  # (method, low_or_a, high_or_None, result)

    def integers(self, low, high=None, *args, **kwargs):
        result = self._rng.integers(low, high, *args, **kwargs)
        self.calls.append(("integers", low, high, result))
        return result

    def uniform(self, low=0.0, high=1.0, *args, **kwargs):
        result = self._rng.uniform(low, high, *args, **kwargs)
        self.calls.append(("uniform", low, high, result))
        return result

    def choice(self, a, *args, **kwargs):
        result = self._rng.choice(a, *args, **kwargs)
        self.calls.append(("choice", a, None, result))
        return result

    def integers_calls(self) -> list[tuple]:
        return [(low, high) for method, low, high, _result in self.calls if method == "integers"]


_CLASS_TRANSFORMS = {
    "random_rotate": lambda mask, rng: random_rotate(mask, rng),
    "random_3directions_rotate": lambda mask, rng: random_3directions_rotate(mask, rng),
    "little_rotate_and_move": lambda mask, rng: little_rotate_and_move(mask, rng),
    "visa_candle": lambda mask, rng: visa_candle(mask, rng),
    "random_reset": lambda mask, rng: random_reset(mask, rng),
}


@pytest.mark.parametrize("name", _CLASS_TRANSFORMS)
def test_class_transform_preserves_shape_dtype_and_binariness(name):
    mask = _two_blob_mask()

    result = _CLASS_TRANSFORMS[name](mask, np.random.default_rng(7))

    assert result.shape == mask.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


@pytest.mark.parametrize("name", _CLASS_TRANSFORMS)
def test_class_transform_is_deterministic_under_a_seed(name):
    mask = _two_blob_mask()
    transform = _CLASS_TRANSFORMS[name]

    first = transform(mask, np.random.default_rng(11))
    again = transform(mask, np.random.default_rng(11))

    assert np.array_equal(first, again)


def test_random_rotate_spins_in_place_without_shifting():
    # The defining difference from little_rotate_and_move: a pure rotation about the mask's centre
    # leaves an off-centre blob's distance from that centre unchanged, however the angle lands.
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[4:10, 4:10] = 255
    image_center = np.array([19.5, 19.5])
    original_radius = np.linalg.norm(np.argwhere(mask == 255).mean(axis=0) - image_center)

    for seed in range(10):
        rotated = random_rotate(mask, np.random.default_rng(seed))
        radius = np.linalg.norm(np.argwhere(rotated == 255).mean(axis=0) - image_center)
        assert radius == pytest.approx(original_radius, abs=2.0)


def test_random_rotate_draws_the_angle_from_the_full_closed_degree_range():
    # rng.integers is high-exclusive, so reaching 360 needs integers(0, 361). A dropped "+1" here
    # would only ever cost the single value 360 -- which renders identically to a 0-degree
    # rotation -- so pin the call bound directly instead of trying to observe it statistically.
    spy = _RecordingRng(np.random.default_rng(0))

    random_rotate(_two_blob_mask(), spy)

    assert spy.integers_calls() == [(0, 361)]


def test_random_rotate_produces_far_more_than_six_distinct_outcomes():
    # Distinguishes it from a random_3directions_rotate body swap, which can only ever produce six
    # outcomes (rotate 90/180/270, two flips, identity) no matter how many seeds are tried.
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[4:10, 4:14] = 255  # asymmetric blob: rotations by different angles rarely coincide

    results = {tuple(random_rotate(mask, np.random.default_rng(seed)).flatten()) for seed in range(100)}

    assert len(results) > 20


def test_three_directions_rotate_returns_one_of_six_outcomes_including_identity():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:6, 9:14] = 255  # asymmetric blob so all six outcomes are distinguishable
    image = Image.fromarray(mask)
    possible_outcomes = [
        mask,
        np.asarray(image.rotate(90), dtype=np.uint8),
        np.asarray(image.rotate(180), dtype=np.uint8),
        np.asarray(image.rotate(270), dtype=np.uint8),
        np.asarray(image.transpose(Image.FLIP_LEFT_RIGHT), dtype=np.uint8),
        np.asarray(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8),
    ]

    results = [random_3directions_rotate(mask, np.random.default_rng(seed)) for seed in range(30)]

    for result in results:
        assert any(np.array_equal(result, outcome) for outcome in possible_outcomes)
    assert any(np.array_equal(result, mask) for result in results)  # identity does occur
    assert any(not np.array_equal(result, mask) for result in results)  # but not always


def test_little_rotate_and_move_moves_the_mask_only_slightly():
    # Distinguishes it from visa_candle/random_reset, which can relocate a component anywhere.
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 45:55] = 255
    original_centroid = np.argwhere(mask == 255).mean(axis=0)
    displacements = []

    for seed in range(15):
        moved = little_rotate_and_move(mask, np.random.default_rng(seed))
        assert set(np.unique(moved)).issubset({0, 255})
        displacements.append(np.linalg.norm(np.argwhere(moved == 255).mean(axis=0) - original_centroid))

    # The default distance=0.05 on a 100px-wide mask caps the shift at int(100*0.05) = 5px.
    assert max(displacements) <= 6.0
    assert max(displacements) > 0  # it does actually move, not a no-op


def test_little_rotate_and_move_draws_angle_and_magnitude_from_exact_bounds():
    # A non-square mask, so a magnitude bound accidentally drawn against height rather than width
    # cannot hide behind height == width, as it would with every other fixture in this file.
    height, width = 100, 256
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[40:60, 118:138] = 255
    spy = _RecordingRng(np.random.default_rng(0))

    little_rotate_and_move(mask, spy, angles=7, distance=0.05, transpose=False)

    calls = spy.integers_calls()
    assert (0, 8) in calls  # angle: integers(0, angles + 1), angles=7
    # magnitude: integers(0, int(width * distance) + 1) = integers(0, int(256 * 0.05) + 1) =
    # integers(0, 13) -- not round(256 * 0.05) + 1 = 14, and not int(height * distance) + 1 = 6.
    assert (0, 13) in calls
    assert (0, 6) not in calls
    assert (0, 14) not in calls


def test_little_rotate_and_move_rotates_both_ways():
    # Distinguishes rng.choice([angle, 360 - angle]) from a mutant that always keeps `angle`,
    # which would make every non-zero rotation land on the same side.
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[20:26, 70:90] = 255  # off-centre and elongated, so its centroid vector has a clear swing
    center = np.array([59.5, 59.5])
    original_vector = np.argwhere(mask == 255).mean(axis=0) - center

    signs = set()
    for seed in range(200):
        # angles>0 with distance=0.0 isolates rotation: magnitude is always drawn from
        # integers(0, int(width * 0.0) + 1) == integers(0, 1) == 0, so translation never applies.
        result = little_rotate_and_move(mask, np.random.default_rng(seed), angles=8, distance=0.0, transpose=False)
        vector = np.argwhere(result == 255).mean(axis=0) - center
        cross = original_vector[0] * vector[1] - original_vector[1] * vector[0]
        if abs(cross) > 1.0:  # skip angle in {0, 360}, which carries no rotation-sign information
            signs.add(cross > 0)

    assert signs == {True, False}


def test_little_rotate_and_move_shift_reaches_exactly_int_width_times_distance():
    # Task's own example: at 256px with distance=0.05, int() gives 12 where round() gives 13.
    height, width = 100, 256
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[45:55, 118:138] = 255  # a small blob near the centre, far from every edge
    distance = 0.05
    expected_bound = int(width * distance)
    center = np.array([(height - 1) / 2, (width - 1) / 2])
    original_vector = np.argwhere(mask == 255).mean(axis=0) - center

    max_abs_shift = 0.0
    for seed in range(300):
        # angles=0 isolates translation: rng.choice([0, 360]) always picks a no-op rotation.
        result = little_rotate_and_move(mask, np.random.default_rng(seed), angles=0, distance=distance)
        vector = np.argwhere(result == 255).mean(axis=0) - center
        displacement = vector - original_vector
        max_abs_shift = max(max_abs_shift, abs(displacement[0]), abs(displacement[1]))

    # Would read ~5 if the bound were mistakenly drawn against height (int(100 * 0.05) = 5) and
    # ~13 if drawn with round() instead of int() (round(256 * 0.05) = 13) -- either a mutant this
    # non-square mask is chosen specifically to expose.
    assert max_abs_shift == pytest.approx(expected_bound, abs=0.5)


def test_little_rotate_and_move_gives_a_180_degree_result_iff_transpose_is_set():
    mask = _two_blob_mask()
    rotated_180 = np.asarray(Image.fromarray(mask).rotate(180), dtype=np.uint8)

    # angles=0, distance=0.0 isolate the transpose coin: with no other rotation or shift active,
    # the only possible outcomes are the original mask or its 180-degree rotation.
    with_transpose = [
        little_rotate_and_move(mask, np.random.default_rng(seed), angles=0, distance=0.0, transpose=True)
        for seed in range(30)
    ]
    without_transpose = [
        little_rotate_and_move(mask, np.random.default_rng(seed), angles=0, distance=0.0, transpose=False)
        for seed in range(30)
    ]

    assert any(np.array_equal(result, mask) for result in with_transpose)
    assert any(np.array_equal(result, rotated_180) for result in with_transpose)
    assert all(np.array_equal(result, mask) for result in without_transpose)
    assert not any(np.array_equal(result, rotated_180) for result in without_transpose)


def test_visa_candle_moves_exactly_one_component_and_leaves_the_other_untouched():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255  # component A
    mask[70:80, 70:80] = 255  # component B

    result = visa_candle(mask, np.random.default_rng(0))

    a_untouched = np.array_equal(result[10:20, 10:20], mask[10:20, 10:20])
    b_untouched = np.array_equal(result[70:80, 70:80], mask[70:80, 70:80])
    assert a_untouched != b_untouched  # exactly one component moved, not both or neither
    # The moved component's own original footprint was erased, not just redrawn in place.
    moved_original_area = mask[70:80, 70:80].sum() if not b_untouched else mask[10:20, 10:20].sum()
    moved_remaining_area = result[70:80, 70:80].sum() if not b_untouched else result[10:20, 10:20].sum()
    assert moved_remaining_area < moved_original_area
    # Total foreground area is unchanged: the piece was relocated, not duplicated or dropped.
    assert result.sum() == mask.sum()


def test_visa_candle_shift_reaches_the_full_range_on_both_axes():
    # A single component, well clear of the canvas edges, so a pure translation measurement is not
    # confounded by clipping or by which of several components got chosen.
    shift_pixels = 5
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[25:35, 25:35] = 255
    center_before = np.argwhere(mask == 255).mean(axis=0)

    dys, dxs = [], []
    for seed in range(250):
        result = visa_candle(mask, np.random.default_rng(seed), shift_pixels=shift_pixels, rotate=False)
        center_after = np.argwhere(result == 255).mean(axis=0)
        dy, dx = center_after - center_before
        dys.append(dy)
        dxs.append(dx)

    assert max(dys) == pytest.approx(shift_pixels, abs=0.5)
    assert min(dys) == pytest.approx(-shift_pixels, abs=0.5)
    assert max(dxs) == pytest.approx(shift_pixels, abs=0.5)
    assert min(dxs) == pytest.approx(-shift_pixels, abs=0.5)


def test_visa_candle_shift_and_rotation_bounds_are_exact():
    mask = _two_blob_mask()
    spy = _RecordingRng(np.random.default_rng(0))

    visa_candle(mask, spy, shift_pixels=5, rotate=True)

    # dy and dx are each drawn from integers(-shift_pixels, shift_pixels + 1); a mutant dropping
    # the "+1" would cost only the single value +shift_pixels, which the shift-range test above
    # would need an impractical sample count to catch statistically -- pin the bound directly.
    assert spy.integers_calls().count((-5, 6)) == 2
    uniform_calls = [(low, high) for method, low, high, _result in spy.calls if method == "uniform"]
    assert uniform_calls == [(-10.0, 10.0)]


def test_random_reset_preserves_the_number_of_connected_components():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    mask[4:8, 30:34] = 255
    mask[40:44, 10:14] = 255
    _, components_before = ndimage.label(mask >= 128)

    result = random_reset(mask, np.random.default_rng(1))

    _, components_after = ndimage.label(result >= 128)
    assert components_before == components_after == 3
    assert result.sum() == mask.sum()  # every component was relocated whole, not clipped
    assert not np.array_equal(result, mask)  # and actually moved


def test_random_reset_relocates_every_component_not_just_one():
    # Distinguishes it from visa_candle, which moves exactly one component and leaves the rest
    # untouched (see test_visa_candle_moves_exactly_one_component_and_leaves_the_other_untouched
    # above) -- the two functions have compatible (mask, rng) signatures, so a body swap between
    # them would otherwise go undetected.
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:8, 4:8] = 255  # component A
    mask[4:8, 30:34] = 255  # component B
    mask[40:44, 50:54] = 255  # component C
    original_boxes = [(4, 4), (4, 30), (40, 50)]

    for seed in range(10):
        result = random_reset(mask, np.random.default_rng(seed))
        untouched = sum(
            np.array_equal(result[y : y + 4, x : x + 4], mask[y : y + 4, x : x + 4]) for y, x in original_boxes
        )
        # visa_candle would structurally leave exactly 2 of the 3 components untouched; genuine
        # random_reset relocates every component (a coincidental round-trip to the same 4x4 spot
        # on a 64x64 canvas is astronomically unlikely).
        assert untouched <= 1


def test_random_reset_relies_on_the_full_100_placement_attempts():
    # A canvas tight enough that a single random placement collides with an already-placed
    # component a substantial fraction of the time, so success is only reliable with the
    # documented 100 retries per component -- not with one attempt.
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:8, 2:8] = 255  # component A: 6x6, placed first, always succeeds into an empty canvas
    mask[12:18, 12:18] = 255  # component B: 6x6, must avoid A
    total_area = int(mask.sum())

    drops = sum(1 for seed in range(60) if int(random_reset(mask, np.random.default_rng(seed)).sum()) != total_area)

    assert drops <= 2  # a 1-attempt mutant drops component B roughly a third of the time


def test_augment_mask_dispatches_the_five_class_transforms(tmp_path: Path):
    mask = _two_blob_mask()
    for mode in _CLASS_TRANSFORMS:
        config = _config(tmp_path, mask_augmentation=mode)

        direct = _CLASS_TRANSFORMS[mode](mask, np.random.default_rng(3))
        via_dispatch = augment_mask(mask, config, np.random.default_rng(3))

        assert np.array_equal(direct, via_dispatch)


def test_augment_mask_passes_little_rotate_and_move_parameters_through(tmp_path: Path):
    config = _config(
        tmp_path,
        mask_augmentation="little_rotate_and_move",
        mask_transform_angles=10,
        mask_transform_distance=0.1,
        mask_transform_transpose=True,
    )
    mask = _two_blob_mask(64)

    direct = little_rotate_and_move(mask, np.random.default_rng(4), angles=10, distance=0.1, transpose=True)
    via_dispatch = augment_mask(mask, config, np.random.default_rng(4))

    assert np.array_equal(direct, via_dispatch)


def test_augment_mask_passes_visa_candle_parameters_through(tmp_path: Path):
    config = _config(tmp_path, mask_augmentation="visa_candle", visa_candle_shift_pixels=5, visa_candle_rotate=True)
    mask = _two_blob_mask(64)

    direct = visa_candle(mask, np.random.default_rng(2), shift_pixels=5, rotate=True)
    via_dispatch = augment_mask(mask, config, np.random.default_rng(2))

    assert np.array_equal(direct, via_dispatch)


@pytest.mark.parametrize("mode", ["rotate_180", "rotate_3_directions", "rotate_all_directions"])
def test_training_augmentation_moves_image_and_mask_together(mode):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[4:12, 4:12] = 200
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:12, 4:12] = 255

    aug_image, aug_mask = augment_training_pair(image, mask, mode, np.random.default_rng(2))

    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape
    # The bright square and the mask square must land in the same place, or the spatial condition
    # would no longer describe the image it is paired with.
    assert np.array_equal(aug_image[..., 0] > 100, aug_mask > 128)


def test_training_augmentation_is_a_no_op_when_disabled():
    image = np.full((8, 8, 3), 7, dtype=np.uint8)
    mask = np.full((8, 8), 255, dtype=np.uint8)

    aug_image, aug_mask = augment_training_pair(image, mask, "none", np.random.default_rng(0))

    assert np.array_equal(aug_image, image)
    assert np.array_equal(aug_mask, mask)


def test_rotate_all_directions_fills_with_the_border_mean_and_black():
    # personalized.py:255-277 fills the rotated image with the mean colour of four 20px border
    # strips and the rotated mask with black.
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 240  # uniform red border and interior -> mean fill is (240, 0, 0)
    mask = np.full((64, 64), 255, dtype=np.uint8)

    aug_image, aug_mask = augment_training_pair(image, mask, "rotate_all_directions", np.random.default_rng(5))

    corner_image, corner_mask = aug_image[0, 0], aug_mask[0, 0]
    assert corner_mask == 0 or np.array_equal(aug_mask, mask)  # black fill unless the angle was 0
    assert corner_image[0] >= 200  # filled from the red border mean, not black


def test_training_augmentation_is_reproducible_under_a_seed():
    image = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:20, 8:20] = 255

    first = augment_training_pair(image, mask, "rotate_all_directions", np.random.default_rng(9))
    again = augment_training_pair(image, mask, "rotate_all_directions", np.random.default_rng(9))

    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])


def test_stored_mask_selection_is_capped_and_reproducible():
    masks = np.stack([np.full((8, 8), value, dtype=np.uint8) for value in range(20)])

    first = select_stored_masks(masks, count=10, rng=np.random.default_rng(3))
    again = select_stored_masks(masks, count=10, rng=np.random.default_rng(3))

    assert first.shape == (10, 8, 8)
    assert np.array_equal(first, again)


def test_stored_mask_selection_handles_fewer_masks_than_requested():
    masks = np.stack([np.zeros((8, 8), dtype=np.uint8) for _ in range(3)])

    assert select_stored_masks(masks, count=10, rng=np.random.default_rng(0)).shape == (3, 8, 8)
