# UCF-Crime and COMMAND

`pyclad.video` implements a clean-room COMMAND recreation using precomputed
RGB/flow features. It uses the same `Model`, `ConceptsDataset`,
`ConceptIncrementalScenario`, `BaseMetric`, and output-writer contracts as
other pyCLAD modalities.

## Package layout

Video follows the same data, model, strategy, callback, and metric organization
as `pyclad.vision`:

- `data/command_ucf_crime.py` contains the dataset and archive reader;
  `data/video_bag_concept.py` and `data/sample.py` hold the video data contracts.
- `models/command/config.py` holds architecture, loss, and training settings;
  `architecture.py` holds the neural networks; `command.py` holds the model,
  composite loss, and complete-bag training/replay implementation.
- `strategies/cont_train.py` connects the model to the core scenario.
- `callbacks/video_frame_metric_callback.py` handles frame evaluation, using
  `metrics/frame_score_utils.py` and the individual frame metric modules.
- `prediction_results.py` holds the modality-specific prediction result.

Import `CommandModel` from `pyclad.video.models.command.command` and its
configuration classes from `pyclad.video.models.command.config`.

## Run the example

From a source checkout:

```shell
pip install -e '.[video]'
python examples/models/video/command_example.py
```

The example composes the dataset, model, strategy, callbacks, and scenario
directly, as the vision examples do. Edit their configuration in the script
to change the run. `CommandUcfCrimeDataset()` downloads the paired feature
archive when needed and reuses its extracted cache. To read an existing local
archive without downloading, replace the dataset assignment with:

```python
dataset = CommandUcfCrimeDataset(root="/path/to/UCF-Crime")
```

To require a previously downloaded Hugging Face cache, use:

```python
dataset = CommandUcfCrimeDataset(local_files_only=True)
```

The script sets 20 epochs per task; `CommandTrainerConfig` defaults to 100.
It selects CUDA when available, then MPS, then CPU. Edit `epochs` and `device`
in the script to choose other settings.

Results are written to `output.json` in the working directory, including
video-level and frame-level ROC-AUC/AP matrices, final-step averages, timing,
and run metadata. To save a model checkpoint, add this after `scenario.run()`:

```python
model.save_checkpoint(pathlib.Path("command-results.pt"))
```

A local archive must have this layout:

```text
UCF-Crime/
├── all_rgbs/
├── all_flows/
├── train_normal.txt
├── train_anomaly.txt
├── test_normalv2.txt
└── test_anomalyv2.txt
```

Each stream contains one finite `(32, 1024)` NumPy array per manifest row,
located at `<stream>/<video-relative-path>.npy`. The reader rejects malformed
features, unbalanced training manifests, duplicate test videos, and train/test
overlap. Anomaly intervals use positive, ordered, one-based inclusive frame
coordinates; endpoints past the video length are clipped. Repeated training
paths retain distinct bag identifiers, preserving
the COMMAND release's 1,620 rows rather than deduplicating its 1,610 videos.

## Continual workflow

The recreated Section IV-C stream groups anomaly classes as follows:

| Task | Classes |
| --- | --- |
| T1 | Abuse, Arrest, Arson, Assault |
| T2 | Burglary, Explosion, Fighting, RoadAccidents |
| T3 | Robbery, Shooting, Shoplifting, Stealing, Vandalism |

Each class receives an equally sized, disjoint shard of the normal training
rows. `read_dataset(max_videos_per_class=1)` limits training for smoke runs;
the complete test split stays fixed after every task.

`VideoBagConcept` keeps each video as one object-array row. Its frame labels
remain outside the feature tensors. `CommandModel.fit()` performs ContTrain++
training, retaining optimizer state and complete-bag replay across tasks;
`ContTrainPlusPlusStrategy` connects it to the core scenario. Prediction
returns bag maxima plus window scores. `VideoFrameMetricCallback` expands
those scores to frames and reuses the core metric-matrix reporting, rejecting
predictions that leave any video's frames uncovered. Checkpoints are saved
explicitly through `model.save_checkpoint(path)`. Load them with
`model.load_checkpoint(path)` using the same configuration; the destination
`device` may differ. New checkpoints restore CPU and CUDA random-generator state
for dropout continuation on the same backend. Loading restores those process-wide
RNG states; identical results across devices are not guaranteed. Older checkpoints
remain loadable but cannot restore Torch RNG state that was never saved.
Checkpoints using the older distance calculation recalibrate from retained
normal bags. If no normal bags remain, an automatic threshold requires fitting
normal bags again; an explicitly configured raw threshold remains usable.

The configured seed controls network initialization and training-bag order.
As with the vision models, constructing a network seeds the process-wide Torch
generator. Passing a prebuilt network preserves its existing weights.

The weakly supervised COMMAND objective and replay operate on complete temporal
bags. Existing tabular strategies such as A-GEM, EWC, and LwF are unchanged;
they are not adapters for this bag-based model.

The example composes the public classes directly; no video-specific CLI or
experiment runner is required. Use `JsonOutputWriter` to serialize the model,
dataset, strategy, and metric callbacks as in other pyCLAD examples.

## Recreation contract

The source is *COMMANDing anomalies: Continual video anomaly detection via
dual-memory and temporal mamba modeling*,
[DOI 10.1016/j.neucom.2026.132943](https://doi.org/10.1016/j.neucom.2026.132943).
Published equations and tables take precedence; public experimental material
informs settings left unspecified. This is not author code or a verified
numerical reproduction.

The architecture retains 32 snippets through RGB/flow fusion, gated temporal
state-space blocks, one-dimensional CBAM, and primary/secondary memory banks.
Defaults use 2,048 feature dimensions, two `256 x 2048` memories, and a
128-dimensional contrastive projection. The default network has 15,036,943
trainable parameters. The objective combines weak MIL ranking, supervised
contrastive, focal classification, anomaly separation, sparsity, smoothness,
and score regularization. Dual-memory deviation supplies the anomaly scores.

Training defaults are 100 epochs per task, batch size 32, replay batch size
16, Adam at `1e-4`, secondary-memory learning rate `1e-5`, decay by 0.1 every
10 epochs, and a 1,000-video replay buffer. Replay balances label, task, and
anomaly-peak quartile. Sampling rotates through retained video IDs within each
cell, and checkpoints preserve the rotation. Older checkpoints without these
cursors start a fresh deterministic replay cycle. The task scheduler restarts
while optimizer and replay state persist.

The paper leaves temporal block count, CBAM settings, state/projection widths,
dropout, differentiated memory rates, novelty threshold, and exact replacement
rules incompletely specified. `CommandArchitectureConfig`, `CommandLossConfig`,
and `CommandTrainerConfig` expose these assumptions; model metadata and
checkpoints retain their values. Tests establish implementation behavior,
not agreement with published benchmark numbers.
