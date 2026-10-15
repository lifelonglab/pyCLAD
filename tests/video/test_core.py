"""Core video data, schema, and frame-evaluation tests."""

from __future__ import annotations

import numpy as np
import pytest

from pyclad.video.metrics.frame_average_precision import FrameAveragePrecision
from pyclad.video.metrics.frame_roc_auc import FrameRocAuc


class TestVideoCore:

    def test_overlapping_window_scores_map_back_to_frames(self):
        from pyclad.video.data.sample import VideoWindow
        from pyclad.video.metrics.frame_score_utils import window_scores_to_frame_scores

        windows = (VideoWindow("video", 0, 2), VideoWindow("video", 2, 4))
        frame_scores = window_scores_to_frame_scores(windows, [0.2, 0.8], {"video": 5})
        np.testing.assert_allclose(frame_scores["video"], [0.2, 0.2, 0.5, 0.8, 0.8])

    def test_frame_metrics_report_perfect_ranking(self):
        labels = np.asarray([0, 0, 0, 1])
        scores = np.asarray([0.0, 0.1, 0.2, 0.9])
        for metric in (FrameRocAuc(), FrameAveragePrecision()):
            assert metric.compute(scores, None, labels) == 1.0
            assert np.isnan(metric.compute(scores, None, np.zeros(4)))

    @pytest.mark.parametrize("metric_class, expected", [(FrameRocAuc, 0.75), (FrameAveragePrecision, 5 / 6)])
    def test_frame_metrics_flatten_temporal_arrays(self, metric_class, expected):
        labels = [[0, 1], [0, 1]]
        scores = [[0.1, 0.2], [0.8, 0.9]]
        metric = metric_class()
        assert metric.compute(scores, None, labels) == pytest.approx(expected)
        assert np.isnan(metric.compute(np.empty((2, 0)), None, np.empty((2, 0))))

    @pytest.mark.parametrize("event_style", ["positional", "keyword"])
    def test_frame_callback_accepts_core_events_and_uses_temporal_predictions(self, event_style):
        from pyclad.callbacks.composite_callback import CallbackComposite
        from pyclad.callbacks.evaluation.concept_metric_evaluation import (
            ConceptMetricCallback,
        )
        from pyclad.data.concept import Concept
        from pyclad.metrics.base.roc_auc import RocAuc
        from pyclad.video.callbacks.video_frame_metric_callback import (
            VideoFrameMetricCallback,
        )
        from pyclad.video.data.sample import VideoWindow
        from pyclad.video.data.video_bag_concept import VideoBagConcept

        labels = np.asarray([0, 1])
        concept = VideoBagConcept(
            name="test", data=np.empty(0), labels=labels, frame_labels={"video": np.asarray([0, 0, 1, 1])}
        )
        core_callback = ConceptMetricCallback(RocAuc(), [])
        frame_callback = VideoFrameMetricCallback(FrameRocAuc())
        callbacks = CallbackComposite([core_callback, frame_callback])
        callbacks.after_training(Concept(name="train", data=np.empty(0)))
        metadata = {
            "window_scores": np.asarray([0.1, 0.9]),
            "windows": (VideoWindow("video", 0, 1), VideoWindow("video", 2, 3)),
        }
        if event_style == "positional":
            callbacks.after_evaluation(concept, labels, labels, np.asarray([0.9, 0.1]), **metadata)
        else:
            callbacks.after_evaluation(
                evaluated_concept=concept,
                y_true=labels,
                y_pred=labels,
                anomaly_scores=np.asarray([0.9, 0.1]),
                **metadata,
            )

        assert core_callback.dense_matrix() == [[0.0]]
        assert frame_callback.dense_matrix() == [[1.0]]

    @pytest.mark.parametrize("event_style", ["positional", "keyword"])
    @pytest.mark.parametrize("missing", ["window_scores", "windows", "concept", "frame_labels"])
    def test_frame_callback_skips_predictions_without_frame_evaluation_data(self, missing, event_style):
        from pyclad.data.concept import Concept
        from pyclad.video.callbacks.video_frame_metric_callback import (
            VideoFrameMetricCallback,
        )
        from pyclad.video.data.sample import VideoWindow
        from pyclad.video.data.video_bag_concept import VideoBagConcept

        callback = VideoFrameMetricCallback(FrameRocAuc())
        concept = VideoBagConcept(name="test", data=np.empty(0), frame_labels={"video": np.asarray([0, 1])})
        prediction = {
            "y_true": np.asarray([0, 1]),
            "y_pred": np.asarray([0, 1]),
            "anomaly_scores": np.asarray([0.1, 0.9]),
            "window_scores": np.asarray([0.1, 0.9]),
            "windows": (VideoWindow("video", 0, 0), VideoWindow("video", 1, 1)),
        }
        if missing == "concept":
            concept = Concept(name="test", data=np.empty(0))
        elif missing == "frame_labels":
            concept.frame_labels = {}
        else:
            prediction.pop(missing)
        callback.after_training(Concept(name="train", data=np.empty(0)))
        if event_style == "positional":
            callback.after_evaluation(
                concept,
                prediction.pop("y_true"),
                prediction.pop("y_pred"),
                prediction.pop("anomaly_scores"),
                **prediction,
            )
        else:
            callback.after_evaluation(evaluated_concept=concept, **prediction)
        assert callback.info() == {}

    def test_frame_callback_reports_incomplete_matrix_after_a_skipped_evaluation(self):
        from pyclad.data.concept import Concept
        from pyclad.video.callbacks.video_frame_metric_callback import (
            VideoFrameMetricCallback,
        )
        from pyclad.video.data.sample import VideoWindow
        from pyclad.video.data.video_bag_concept import VideoBagConcept

        callback = VideoFrameMetricCallback(FrameRocAuc())
        concept = VideoBagConcept(name="test", data=np.empty(0), frame_labels={"video": np.asarray([0, 1])})
        callback.after_training(Concept(name="T1", data=np.empty(0)))
        callback.after_evaluation(
            evaluated_concept=concept,
            window_scores=np.asarray([0.1, 0.9]),
            windows=(VideoWindow("video", 0, 0), VideoWindow("video", 1, 1)),
        )
        assert callback.dense_matrix() == [[1.0]]
        callback.after_training(Concept(name="T2", data=np.empty(0)))
        callback.after_evaluation(evaluated_concept=concept)
        with pytest.raises(ValueError, match="incomplete"):
            callback.info()

    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    @pytest.mark.parametrize("missing", ["frame", "video"])
    def test_frame_metrics_reject_incomplete_prediction_coverage(self, aggregation, missing):
        from pyclad.video.data.sample import VideoWindow
        from pyclad.video.metrics.frame_score_utils import window_scores_to_frame_scores

        windows = (VideoWindow("video", 0, 1), VideoWindow("video", 3, 4))
        frame_counts = {"video": 5}
        if missing == "video":
            windows = (VideoWindow("video", 0, 2), VideoWindow("video", 3, 4))
            frame_counts["unpredicted"] = 5
        with pytest.raises(ValueError, match="do not cover all frames"):
            window_scores_to_frame_scores(windows, [0.2, 0.8], frame_counts, aggregation=aggregation)
