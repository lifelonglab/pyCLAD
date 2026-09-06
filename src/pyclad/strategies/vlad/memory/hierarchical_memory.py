import logging
import math
from typing import Any, Dict, FrozenSet, Optional, Tuple

import numpy as np

from pyclad.output.output_writer import InfoProvider
from pyclad.strategies.vlad.distance import (
    ExactWassersteinDistance,
    SlicedWassersteinDistance,
    WassersteinDistance,
)
from pyclad.strategies.vlad.memory.concept_node import ConceptNode
from pyclad.strategies.vlad.memory.summarization import kmeans_summarize, pyramid_budget

logger = logging.getLogger(__name__)


def _distance_ratio(distance: float, threshold: float) -> float:
    """`distance / threshold`, guarding against a threshold of exactly 0 (a degenerate, perfectly
    tight concept): matches only an exact (0-distance) hit, instead of always matching (division
    by zero) or never matching (naive truthiness treats `0.0` as falsy)."""
    if threshold > 0:
        return distance / threshold
    return 0.0 if distance == 0 else math.inf


class VladMemory(InfoProvider):
    """VLAD's hierarchical memory: concept storage, Consolidation (Eq. 3-4), threshold
    calibration (Eq. 2), and Pyramidal + within-concept Memory Summarization (Eq. 5-8).

    Single source of truth for every concept's samples and threshold — change point detectors
    (see :mod:`pyclad.strategies.vlad.change_point_detection`) read and mutate concept data
    through this class rather than keeping their own copy, so e.g. summarization is immediately
    visible to future comparisons.

    :param memory_bound: soft upper bound on total stored samples (`M_B`).
    :param summarization_trigger_ratio: summarization runs once total samples exceed
        `memory_bound * summarization_trigger_ratio` (`M_f`).
    :param subconcept_threshold_ratio: looser multiplier applied to an existing concept's own
        threshold when deciding whether a new distribution should become its sub-concept during
        Consolidation (`W_epsilon`).
    :param mini_batch_size: size of the point-cloud chunks used both to calibrate a concept's
        threshold (Eq. 2) and by `WassersteinChangePointDetector` to scan incoming data
        (`C_omega`) — that detector reads this value off `memory` rather than taking its own
        copy, so the two can never disagree.
    :param threshold_ratio: multiplier applied to a concept's largest internal chunk distance to
        obtain its admission threshold (`C_epsilon`).
    :param min_distribution_size: minimum samples a concept must accumulate before its threshold
        is (first) calibrated (`C_kappa`). Must be at least `2 * mini_batch_size`, otherwise the
        threshold trivially calibrates to 0 (see `recompute_threshold`).
    :param n_centroids: number of k-means clusters used for within-concept summarization (`M_k`).
    :param max_comparison_size: every Wasserstein distance computed by this class subsamples
        either side down to at most this many points first. `ExactWassersteinDistance`'s
        optimal-transport solve is empirically closer to O(n^3) than linear, so comparing two
        large point clouds directly — which can happen here even though the paper/reference
        implementation always keeps one side small — is intractable without this bound (raise it,
        or use `SlicedWassersteinDistance`, if that cost matters to you).
        Default 128 keeps one comparison well under a second; raise it only if concepts are few
        and need finer-grained distinguishing.
    :param distance: the Wasserstein distance implementation to use for every comparison —
        defaults to :class:`~pyclad.strategies.vlad.distance.ExactWassersteinDistance` (what the
        paper/reference implementation use). The much faster, approximate
        :class:`~pyclad.strategies.vlad.distance.SlicedWassersteinDistance` is validated as
        reliable only against large point clouds (whole concept-incremental batches); it produced
        an inconsistent concept count when compared against `mini_batch_size`-sized chunks
        (concept-agnostic scanning) in testing — see its docstring, and prefer
        `for_concept_incremental`/`for_concept_agnostic` over picking this by hand.
    :param rng: source of randomness for k-means summarization and comparison subsampling.
    :raises ValueError: if `min_distribution_size < 2 * mini_batch_size`.
    """

    def __init__(
        self,
        memory_bound: int,
        summarization_trigger_ratio: float = 5.0,
        subconcept_threshold_ratio: float = 5.0,
        mini_batch_size: int = 4,
        threshold_ratio: float = 2.0,
        min_distribution_size: int = 1024,
        n_centroids: int = 5,
        max_comparison_size: int = 128,
        distance: Optional[WassersteinDistance] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if min_distribution_size < 2 * mini_batch_size:
            raise ValueError(
                f"min_distribution_size ({min_distribution_size}) must be at least "
                f"2 * mini_batch_size ({2 * mini_batch_size}): threshold calibration (Eq. 2) "
                f"needs at least 2 non-overlapping chunks, otherwise the only 'chunk' IS the "
                f"whole buffer and the threshold trivially calibrates to 0."
            )
        self.memory_bound = memory_bound
        self.summarization_trigger_ratio = summarization_trigger_ratio
        self.subconcept_threshold_ratio = subconcept_threshold_ratio
        self.mini_batch_size = mini_batch_size
        self.threshold_ratio = threshold_ratio
        self.min_distribution_size = min_distribution_size
        self.n_centroids = n_centroids
        self.max_comparison_size = max_comparison_size
        self._wasserstein_distance = distance if distance is not None else ExactWassersteinDistance()
        if rng is None:
            logger.info(
                "VladMemory: no `rng` was provided; results will not be reproducible across "
                "runs. Pass rng=np.random.default_rng(seed) for reproducible experiments."
            )
        self._rng = rng if rng is not None else np.random.default_rng()

        self._nodes: Dict[str, ConceptNode] = {}
        self._next_id = 0
        self._bounded_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._warned_no_calibration = False

    @classmethod
    def for_concept_agnostic(cls, memory_bound: int, **kwargs) -> "VladMemory":
        """`VladMemory` for use with `WassersteinChangePointDetector` (concept-agnostic mode).
        Defaults `distance` to `ExactWassersteinDistance()` — the only option validated as
        reliable against `mini_batch_size`-sized chunk comparisons. Pass `distance=...` in
        `kwargs` to override anyway. All other parameters are `VladMemory.__init__`'s.
        """
        kwargs.setdefault("distance", ExactWassersteinDistance())
        return cls(memory_bound, **kwargs)

    @classmethod
    def for_concept_incremental(cls, memory_bound: int, **kwargs) -> "VladMemory":
        """`VladMemory` for use with `BoundaryChangePointDetector` (concept-incremental mode).
        Defaults `distance` to `SlicedWassersteinDistance()` — validated as reliably matching
        `ExactWassersteinDistance`'s concept count while being far faster, since this mode only
        ever compares whole (large) concept batches, not tiny chunks. Pass `distance=...` in
        `kwargs` to override anyway. All other parameters are `VladMemory.__init__`'s.
        """
        kwargs.setdefault("distance", SlicedWassersteinDistance())
        return cls(memory_bound, **kwargs)

    def create_concept(self, samples: np.ndarray) -> str:
        """Register a brand-new concept seeded with `samples`, place it in the hierarchy via
        Consolidation (Eq. 3-4), and calibrate its threshold if enough data is already available.
        """
        samples = np.asarray(samples)
        concept_id = self._fresh_id()
        parent_id, layer = self._place_in_hierarchy(samples)
        self._nodes[concept_id] = ConceptNode(
            id=concept_id, parent_id=parent_id, layer=layer, samples=samples, threshold=None
        )
        self._try_calibrate(concept_id)
        return concept_id

    def add_to_concept(self, concept_id: str, samples: np.ndarray) -> None:
        """Extend an existing concept's stored buffer with newly observed (or recurring) data."""
        node = self._nodes[concept_id]
        node.samples = np.concatenate([node.samples, np.asarray(samples)])
        self._try_calibrate(concept_id)

    def recompute_threshold(self, concept_id: str) -> float:
        """Recompute a concept's admission threshold from its current buffer (Eq. 2): re-chunk
        the buffer into `mini_batch_size`-sized groups, and set the threshold to `threshold_ratio`
        times the largest group-to-whole-buffer distance.
        """
        node = self._nodes[concept_id]
        reference = self._bounded_samples(concept_id)
        distances = [
            self._wasserstein_distance(node.samples[i : i + self.mini_batch_size], reference)
            for i in range(0, len(node.samples) - self.mini_batch_size + 1, self.mini_batch_size)
        ]
        threshold = (max(distances) if distances else 0.0) * self.threshold_ratio
        node.threshold = threshold
        return threshold

    def distance_to(self, concept_id: str, candidate: np.ndarray) -> float:
        """Wasserstein distance from `candidate` to a concept's current buffer."""
        reference = self._bounded_samples(concept_id)
        return self._wasserstein_distance(self._bounded(np.asarray(candidate)), reference)

    def find_best_match(
        self, candidate: np.ndarray, exclude: FrozenSet[str] = frozenset()
    ) -> Optional[Tuple[str, float]]:
        """Best (lowest-ratio) calibrated-concept match for `candidate`, per Algorithm 2: eligible
        iff `distance(candidate, concept) / concept.threshold < 1`. `None` if nothing matches.
        """
        candidate = self._bounded(np.asarray(candidate))
        return self._search_calibrated(candidate, threshold_scale=1.0, exclude=exclude)

    def calibrated_concepts(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """Concepts whose threshold has calibrated (no longer "still forming") — the searchable
        pool for change-point/recurrence matching, and the sole *bulk* accessor for concepts'
        samples. For a single, possibly-uncalibrated concept, use `samples()`/`concept_ids()`."""
        return {cid: (node.samples, node.threshold) for cid, node in self._nodes.items() if node.threshold is not None}

    def concept_ids(self) -> FrozenSet[str]:
        """Every concept id currently in memory, calibrated or not."""
        return frozenset(self._nodes.keys())

    def samples(self, concept_id: str) -> np.ndarray:
        """A single concept's current stored buffer, calibrated or not."""
        return self._nodes[concept_id].samples

    def threshold(self, concept_id: str) -> Optional[float]:
        return self._nodes[concept_id].threshold

    def should_summarize(self) -> bool:
        return self.total_samples() > self.memory_bound * self.summarization_trigger_ratio

    def summarize(self) -> None:
        """Pyramidal + within-concept Memory Summarization (Eq. 5-8): shrink every concept's
        buffer to its pyramid-allocated budget via k-means coreset selection, then recalibrate
        the thresholds of concepts that were already calibrated (their buffer content changed).
        """
        budgets = pyramid_budget({cid: node.layer for cid, node in self._nodes.items()}, self.memory_bound)
        for concept_id, node in self._nodes.items():
            node.samples = kmeans_summarize(
                node.samples, budget=budgets[concept_id], n_centroids=self.n_centroids, rng=self._rng
            )
            if node.threshold is not None:
                self.recompute_threshold(concept_id)

    def replay_buffer(self) -> np.ndarray:
        """Union of every concept's current samples (Eq. 9)."""
        buffers = [node.samples for node in self._nodes.values() if len(node.samples) > 0]
        return np.concatenate(buffers) if buffers else np.empty((0,))

    def n_concepts(self) -> int:
        return len(self._nodes)

    def total_samples(self) -> int:
        return sum(len(node.samples) for node in self._nodes.values())

    def name(self) -> str:
        return "VladMemory"

    def info(self) -> Dict[str, Any]:
        return {"name": self.name(), **self.additional_info()}

    def additional_info(self) -> Dict[str, Any]:
        return {
            "memory_bound": self.memory_bound,
            "summarization_trigger_ratio": self.summarization_trigger_ratio,
            "subconcept_threshold_ratio": self.subconcept_threshold_ratio,
            "mini_batch_size": self.mini_batch_size,
            "threshold_ratio": self.threshold_ratio,
            "min_distribution_size": self.min_distribution_size,
            "n_centroids": self.n_centroids,
            "max_comparison_size": self.max_comparison_size,
            "distance": self._wasserstein_distance.info(),
            "n_concepts": self.n_concepts(),
            "n_calibrated_concepts": len(self.calibrated_concepts()),
            "total_samples": self.total_samples(),
        }

    def _try_calibrate(self, concept_id: str) -> None:
        node = self._nodes[concept_id]
        if node.threshold is None and len(node.samples) >= self.min_distribution_size:
            self.recompute_threshold(concept_id)
        self._warn_if_nothing_ever_calibrates()

    def _warn_if_nothing_ever_calibrates(self) -> None:
        """One-time diagnostic for the min_distribution_size silent-failure mode: memory has
        clearly outgrown it in aggregate, yet no single concept individually reached it."""
        if self._warned_no_calibration or self.calibrated_concepts():
            return
        if self.total_samples() > self.min_distribution_size:
            logger.warning(
                "VladMemory has accumulated %d samples across %d concept(s), but none has "
                "calibrated a threshold yet (min_distribution_size=%d). Uncalibrated concepts "
                "can never be matched as recurring, so every new segment will be treated as "
                "brand-new. Consider lowering min_distribution_size below your typical "
                "per-concept sample count.",
                self.total_samples(),
                self.n_concepts(),
                self.min_distribution_size,
            )
            self._warned_no_calibration = True

    def _place_in_hierarchy(self, samples: np.ndarray) -> Tuple[Optional[str], int]:
        """Consolidation (Eq. 3-4): find the nearest already-calibrated concept under a looser,
        `subconcept_threshold_ratio`-scaled threshold. If found, the new distribution becomes its
        sub-concept; otherwise it becomes a new root concept.
        """
        samples = self._bounded(samples)
        match = self._search_calibrated(samples, threshold_scale=self.subconcept_threshold_ratio, exclude=frozenset())
        if match is None:
            return None, 1
        best_id, _ = match
        return best_id, self._nodes[best_id].layer + 1

    def _search_calibrated(
        self, candidate: np.ndarray, threshold_scale: float, exclude: FrozenSet[str]
    ) -> Optional[Tuple[str, float]]:
        """Shared by `find_best_match` (`threshold_scale=1.0`) and `_place_in_hierarchy`
        (`threshold_scale=subconcept_threshold_ratio`): the concept with the smallest
        `distance / (concept.threshold * threshold_scale)` ratio wins, among those `< 1`.
        `candidate` is assumed already bounded by the caller.
        """
        best: Optional[Tuple[str, float]] = None
        for concept_id, (_, threshold) in self.calibrated_concepts().items():
            if concept_id in exclude:
                continue
            distance = self._wasserstein_distance(candidate, self._bounded_samples(concept_id))
            ratio = _distance_ratio(distance, threshold * threshold_scale)
            if ratio < 1 and (best is None or ratio < best[1]):
                best = (concept_id, ratio)
        return best

    def _bounded(self, samples: np.ndarray) -> np.ndarray:
        """Subsample `samples` to `max_comparison_size` points, if larger. For one-off arrays not
        tied to a stored concept; for a concept's own buffer, use `_bounded_samples` (cached)."""
        if len(samples) <= self.max_comparison_size:
            return samples
        indices = self._rng.choice(len(samples), size=self.max_comparison_size, replace=False)
        return samples[indices]

    def _bounded_samples(self, concept_id: str) -> np.ndarray:
        """Bounded view of a concept's buffer, cached and only redrawn when `node.samples` is
        rebound to a new array (checked by identity) — so repeated queries against an unchanged
        concept get a stable subsample instead of independently-random ones each time."""
        node = self._nodes[concept_id]
        if len(node.samples) <= self.max_comparison_size:
            return node.samples
        cached = self._bounded_cache.get(concept_id)
        if cached is not None and cached[0] is node.samples:
            return cached[1]
        indices = self._rng.choice(len(node.samples), size=self.max_comparison_size, replace=False)
        subsample = node.samples[indices]
        self._bounded_cache[concept_id] = (node.samples, subsample)
        return subsample

    def _fresh_id(self) -> str:
        concept_id = f"concept-{self._next_id}"
        self._next_id += 1
        return concept_id
