# Examples

The [examples directory](https://github.com/lifelonglab/pyCLAD/tree/main/examples) holds runnable scripts, one per
feature. Run them from a clone of the repository. Scripts that use a built-in benchmark download it on first use;
the image examples need the datasets described in [Image anomaly detection](guides/vision.md#datasets).

[Your first experiment](getting_started.md) is the best place to start: a complete script, explained step by step.

## Scenarios

- [concept_incremental_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/concept_incremental_example.py):
  four synthetic concepts drawn from different normal distributions, Cumulative strategy and Isolation Forest, in the
  concept-incremental scenario.
- [concept_aware_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/concept_aware_example.py):
  the NSL-KDD scenario stored in `examples/resources`, ReplayEnhanced strategy and Isolation Forest, in the
  concept-aware scenario.
- [concept_agnostic_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/concept_agnostic_example.py):
  three synthetic concepts with anomalies, Cumulative strategy and One-Class SVM, in the concept-agnostic scenario. It
  also measures memory usage with `MemoryUsageCallback`.

## Datasets

- [unsw_dataset_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/unsw_dataset_example.py):
  the UNSW-NB15 benchmark with a PyOD model and the ReplayEnhanced strategy.
- [cad_cicids2017_dataset_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/cad_cicids2017_dataset_example.py):
  the CAD-CICIDS2017 benchmark, one of the tabular CAD benchmarks with predefined orderings.
- [energy_dataset_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/energy_dataset_example.py):
  the Energy Plants benchmark with an autoencoder trained by the EWC strategy.
- [energy_timeseries_dataset_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/energy_timeseries_dataset_example.py):
  the Energy Plants benchmark turned into overlapping windows, scored by LOF through `FlattenTimeSeriesAdapter`.
- [timeseries_ad_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/timeseries_ad_example.py):
  the same windowing on synthetic series whose last steps are anomalous.
- [step_schedule_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/step_schedule_example.py):
  merges consecutive concepts into larger training steps and evaluates with the schedule-aware metrics.

## Models

All in [examples/models](https://github.com/lifelonglab/pyCLAD/tree/main/examples/models); they need the `torch`
extra.

- `autoencoder_torch_example.py` and `variational_autoencoder_torch_example.py`: an autoencoder and a variational
  autoencoder wrapped with `TorchModelAdapter`, trained with the ReplayEnhanced strategy.
- `gru_tae_torch_example.py`, `lstm_tae_torch_example.py`, `tcn_tae_torch_example.py`: temporal autoencoders and
  their variational versions with GRU, LSTM and TCN layers.
- `carla_energy_example.py`: the CARLA contrastive model on the Energy Plants benchmark.

## Strategies

All in [examples/strategies](https://github.com/lifelonglab/pyCLAD/tree/main/examples/strategies).

- `derpp_example.py`: DER++ with an autoencoder, in the concept-agnostic scenario.
- `vlad_concept_agnostic_example.py`: VLAD as published, detecting concept changes on its own.
- `vlad_concept_incremental_example.py`: VLAD using the scenario's concept boundaries.
- `replaycad_example.py`: ReplayCAD, diffusion-based generative replay for images, with PatchCore. Needs the
  `replaycad` extra.

## Image anomaly detection

In [examples/models/vision](https://github.com/lifelonglab/pyCLAD/tree/main/examples/models/vision) and the
examples root; they need the `vision` extra and the datasets described in
[Image anomaly detection](guides/vision.md#datasets).

- `patchcore_torch_example.py`, `fastflow_torch_example.py`, `rd4ad_torch_example.py`, `paste_torch_example.py`:
  one image model each, with the ReplayEnhanced strategy and image- and pixel-level metrics.
- `ucad_torch_example.py`: UCAD with its own strategy. Its results are not comparable to the paper's tables; see
  "Data leakage" in [Vision](vision.md#data-leakage).
- `continual_mega_example.py`: the Continual-MEGA benchmark with a supervised model.
- [multi_dataset_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/multi_dataset_example.py):
  one continual stream of 27 categories drawn from five image datasets (InCLAD-MD).
- [step_schedule_vision_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/step_schedule_vision_example.py)
  and [step_schedule_multi_dataset_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/step_schedule_multi_dataset_example.py):
  CDAD-style step schedules on MVTec AD and on the multi-dataset stream.

## Analysis

- [plot_heatmap_example.py](https://github.com/lifelonglab/pyCLAD/blob/main/examples/plot_heatmap_example.py):
  plots the metric matrix saved in an output JSON file as a heatmap.
