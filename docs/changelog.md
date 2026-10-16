# Changelog

## Unreleased

### Breaking changes

- **`predict()` returns `PredictionResults`** instead of a `(y_pred, anomaly_scores)` tuple, for models and strategies.
  Custom models and strategies must return `PredictionResults(y_pred=..., anomaly_scores=...)`, and code that unpacks
  the tuple must read the two fields instead.
- **`ConceptMetricCallback(metrics=...)` is now `summarized_metrics=...`**, and `ConceptLevelMetric` is renamed to
  `SummarizedMetric`. Metrics that report one value per step (`ForgettingMeasure`) go in the new
  `stepwise_metrics` argument.
- **`Autoencoder` is a `TorchBackbone`**, no longer a PyTorch Lightning `Model`. Wrap it as
  `TorchModelAdapter(Autoencoder(...), StandardRunner(max_epochs=...), batch_size=...)`; the number of epochs moved to
  the runner. See [Models](models.md#pytorch-models).
- **Square-only continual metrics reject rectangular matrices** with `ValueError` instead of computing a number.
- **Undefined metric values are `NaN`, and the metric callbacks stop the run by default** (`on_undefined="raise"`)
  when the base metric cannot be computed, for example ROC-AUC on a test concept with a single class. Pass
  `on_undefined="propagate"` to finish the run and record `NaN`. See [Metrics](metrics.md#undefined-values).
- **The output JSON key `tran_concepts_no` is now `train_concepts_no`.** Scripts that parse saved results need
  updating. `NaN` values are written as `null`.
- **Python 3.10 or newer is required** (previously 3.8).
- PyTorch and PyTorch Lightning are now in the `torch` extra (`pip install "pyclad[torch]"`), and the image models
  in the `vision`, `ucad` and `replaycad` extras. See [Installation](learn/installation.md).

### Added

- Strategies: EWC (offline and online), LwF, DER++, A-GEM and VLAD (concept-agnostic and concept-incremental).
- Models: `VariationalAutoencoder`, the `TorchBackbone` / `TorchRunner` / `TorchModelAdapter` split with
  `StandardRunner` and `EarlyStopping`, CARLA, and `FlattenTimeSeriesAdapter` with time-series windowing.
- Datasets: the tabular CAD benchmarks (CAD-CICIDS2017, CAD-CICIDS2018, CAD-CICUNSW, CAD-TCM, CAD-MiniBooNE,
  CAD-Scania, MCAD-CIC-3x1, MCAD-CIC-3xN) with predefined orderings.
- Step schedules: `apply_step_schedule` merges consecutive concepts into larger training steps, with
  `ScheduleAwareConceptMetricCallback` and the schedule-aware metrics for the resulting rectangular matrix.
- Metrics: `AveragePrecision`, `F1Score`, `ForgettingMeasure`, `FinalStepAverage`, `FinalStepForgettingMeasure`, and
  `GroupedConceptMetricCallback`.
- Image anomaly detection in `pyclad.vision`: PatchCore, FastFlow, RD4AD, PaSTe, UCAD, ReplayCAD, the
  Continual-MEGA benchmark, multi-dataset streams, readers for MVTec AD, VisA, MPDD, BTAD and DAGM, and pixel-level
  metrics. See [Vision](vision.md).
- `set_seed` seeds Python, NumPy and PyTorch in one call, and saves the seed with the results.
- `SupervisedModel`, `SupervisedStrategy` and `SupervisedConceptIncrementalScenario` for models trained on labels.

### Fixed

- Scenarios call `after_scenario` on their callbacks also when a run fails, so energy and memory trackers are
  released.
- The heatmap plot orders concepts correctly.

## 0.3.0

See the [release notes on GitHub](https://github.com/lifelonglab/pyCLAD/releases).
