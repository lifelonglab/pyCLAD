# Vision models — quick start

Continual **visual anomaly detection** models for pyCLAD. Each model trains on *normal* images only and produces per-pixel anomaly maps.

Every model ships with a ready-to-run example under `examples/models/vision/`.

## Models

| Model | What it is | Example script |
|---|---|--|
| **PaSTe** | Student–teacher distillation | `paste_torch_example.py` |
| **FastFlow** | Normalizing flow on features | `fastflow_torch_example.py` |

## Setup — extra libraries required

Beyond a normal pyCLAD install, the vision models need **exactly three additional packages** (the deep-learning stack):

| Package | Used for |
|---|---|
| `torch` | networks, tensors, training |
| `torchvision` | pretrained backbones (ResNet / MobileNet / EfficientNet) + feature extraction |
| `pytorch-lightning` | training loop (`pl.Trainer`, `LightningModule`) |

## Datasets

The example scripts read data from `examples/resources/vision/<dataset>/`. Datasets are *not* included in the repository, so after cloning you must download and place them yourself.

**1. Put the dataset here** (folder name is up to you. It just has to match the `root=` in the script):

```
examples/resources/vision/
├── BTech_Dataset_transformed/   # benchmark="btech"
├── mvtec_ad/                    # benchmark="mvtec"
├── MPDD/                        # benchmark="mpdd"
├── VisA/                        # benchmark="visa"
└── DAGM_KaggleUpload/           # benchmark="dagm"
```

**2. Expected layout inside a dataset** — one folder per category, each split into `train` / `test` / `ground_truth`:

```
<dataset>/
└── <category>/                  
    ├── train/<normal>/*.png     # normal images for training
    ├── test/<normal>/*.png      # normal test images
    ├── test/<defect>/*.png      # anomalous test images
    └── ground_truth/<defect>/*_mask.png   # pixel masks for the anomalies
```

The `benchmark=` name tells the reader which naming convention to expect (e.g. BTech uses `ok`/`ko`, MVTec uses `good`), so you only need to drop the dataset in with its original structure.

**3. Where to get the data** (public industrial anomaly-detection datasets):

| Benchmark | Source |
|---|---|
| BTech (BTAD) | https://www.kaggle.com/datasets/thtuan/btad-beantech-anomaly-detection/ |
| MVTec AD | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| VisA | https://github.com/amazon-science/spot-diff |
| MPDD | https://github.com/stepanje/MPDD |
| DAGM 2007 | https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection |

### Loading a dataset in code

The example scripts build their dataset with a single call to `read_vision_dataset` — this is where you point pyCLAD at your data and choose the image size:

```python
import pathlib
from pyclad.vision.data.readers.vision_reader import read_vision_dataset

dataset = read_vision_dataset(
    root=pathlib.Path("../../resources/vision/BTech_Dataset_transformed"),  # dataset folder
    benchmark="btech",         # naming convention: btech | mvtec | mpdd | visa | dagm
    resize_to=(256, 256),      # all images resized to this H×W (use 224 for PaSTe)
    data_mode="numpy",         # "numpy" = load into memory, "paths" = load lazily
    color_mode="rgb",          # "rgb" or "grayscale"
    # max_train_samples_per_category=150,   # uncomment → fewer images per category
    # max_test_samples_per_category=150,
)
```

## Run a model

The examples use **relative paths** to the dataset, so run them from their own folder:

```bash
cd examples/models/vision

python paste_torch_example.py
python fastflow_torch_example.py
```

Each run:

1. loads the dataset as a continual stream of concepts (categories),
2. trains the model on each category in turn (replay strategy),
3. evaluates image- and pixel-level metrics (ROC-AUC, F1, AUPRO, IoU, Dice, …),
4. writes results to **`output.json`** in the current folder.

## Available metrics

The examples evaluate every concept with both **image-level** and **pixel-level** metrics. You attach each one as a callback:
image-level via `ConceptMetricCallback`, pixel-level via `VisionPixelConceptMetricCallback` (needs ground-truth masks).

### Image-level

Computed on the per-image anomaly score. Import from `pyclad.metrics.base`:

| Metric | Class | Import | Measures |
|---|---|---|---|
| ROC-AUC | `RocAuc` | `pyclad.metrics.base.roc_auc` | ranking of normal vs anomalous images |
| F1-Score | `F1Score` | `pyclad.metrics.base.f1_score` | precision/recall balance at the decision threshold |
| Average Precision | `AveragePrecision` | `pyclad.metrics.base.average_precision` | area under the precision–recall curve |

### Pixel-level (vision-specific)

Computed on the per-pixel anomaly map vs the ground-truth mask. Import from `pyclad.vision.metrics`:

| Metric | Class | Import | Measures |
|---|---|---|---|
| Pixel ROC-AUC | `PixelRocAuc` | `pyclad.vision.metrics.pixel_roc_auc` | pixel-wise ranking of anomaly scores *(threshold-free)* |
| Pixel AP | `PixelAveragePrecision` | `pyclad.vision.metrics.pixel_average_precision` | pixel-wise precision–recall area *(threshold-free)* |
| Pixel AUPRO | `PixelAUPRO` | `pyclad.vision.metrics.pixel_aupro` | per-region overlap vs FPR *(threshold-free)* |
| Pixel F1 | `PixelF1Score` | `pyclad.vision.metrics.pixel_f1_score` | F1 of the binarized anomaly map vs mask |
| Pixel IoU | `PixelIoU` | `pyclad.vision.metrics.pixel_iou` | intersection-over-union of predicted vs true defect region |
| Pixel Dice | `PixelDiceScore` | `pyclad.vision.metrics.pixel_dice_score` | Dice overlap of predicted vs true defect region |

### Continual summaries

On top of any base metric you can wrap continual summaries (from `pyclad.metrics.continual`) that aggregate results across the concept sequence: `ContinualAverage`, `BackwardTransfer`, `ForwardTransfer`.

```python
summarized_metrics = [ContinualAverage(), BackwardTransfer(), ForwardTransfer()]

callbacks = [
    # image-level
    ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=summarized_metrics),
    ConceptMetricCallback(base_metric=F1Score(), summarized_metrics=summarized_metrics),
    ConceptMetricCallback(base_metric=AveragePrecision(), summarized_metrics=summarized_metrics),
    # pixel-level (need masks)
    VisionPixelConceptMetricCallback(base_metric=PixelRocAuc(), summarized_metrics=summarized_metrics),
    VisionPixelConceptMetricCallback(base_metric=PixelAUPRO(), summarized_metrics=summarized_metrics),
    # ... PixelAveragePrecision, PixelF1Score, PixelIoU, PixelDiceScore
]
```

## PatchCore

PatchCore is a memory-bank detector: a frozen, pretrained backbone extracts mid-level patch
features (`layer2` + `layer3`) for every training image, a greedy coreset subsamples them into a
small memory bank, and each test patch is scored by nearest-neighbour distance to that bank. No
gradient training happens — `fit()` only extracts features and builds the memory bank.

The image-level score is the maximum over **raw** patch distances; the segmentation map is those
same distances bilinearly upsampled to `input_size` and Gaussian-smoothed with `smoothing_sigma`.
Because the image score is taken before smoothing, it is generally **larger** than the maximum of
the returned map — see divergence 6 below if you're comparing against ADer's published numbers.

### Config (`PatchCoreConfig`)

Defaults match the reference implementation behind ReplayCAD's PatchCore benchmark row (ADer's
`model/patchcore.py`), constant-for-constant.

| Field | Default | Meaning |
|---|---|---|
| `input_size` | `(256, 256)` | Images are resized to this H×W before feature extraction |
| `backbone_name` | `"wide_resnet50_2"` | Any torchvision backbone supported by the shared feature extractor |
| `backbone_return_nodes` | `None` | Explicit override; `None` picks `layer2`/`layer3` automatically |
| `pretrained_backbone` | `True` | Load pretrained ImageNet weights |
| `freeze_backbone` | `True` | Backbone parameters are never updated (PatchCore never trains it) |
| `pretrained_weights` | `"IMAGENET1K_V1"` | Not torchvision's `DEFAULT` (`IMAGENET1K_V2`) — the reference implementation loads the V1 checkpoint |
| `pretrain_embed_dimension` | `1024` | Per-layer patch feature dimension after mean-mapping |
| `target_embed_dimension` | `1024` | Final aggregated patch embedding dimension |
| `patchsize` | `3` | Side length (in feature-map cells) of each patch |
| `patchstride` | `1` | Stride between patches |
| `coreset_sampling_ratio` | `0.1` | Fraction of training patches kept in the memory bank |
| `coreset_projection_dimension` | `128` | Random projection dimension used by the greedy coreset selector |
| `coreset_starting_points` | `10` | Number of random seed points for the greedy coreset search |
| `n_neighbors` | `1` | Neighbours averaged for the nearest-neighbour patch distance |
| `smoothing_sigma` | `4.0` | Gaussian smoothing sigma applied to the segmentation map only |

### Running the example

```bash
cd examples/models/vision
python patchcore_torch_example.py
```

Like the other examples, this reads a dataset via `read_vision_dataset` (see
[Datasets](#datasets) above), wraps the model in a replay strategy, attaches the same image- and
pixel-level metric callbacks, and writes results to `output.json`.

## ReplayCAD

ReplayCAD (Hu et al., *ReplayCAD: Generative Diffusion Replay for Continual Anomaly Detection*,
IJCAI 2025, arXiv:2505.06603) is a generative-replay strategy: instead of storing real images from
earlier concepts, it compresses each one into a small, frozen-diffusion-model conditioning (a
learned semantic embedding plus a mask-projection MLP) that can regenerate representative images
on demand, and replays those alongside new data when refitting the detector.

`ReplayCADStrategy(model, memory)` is detector-agnostic — `model` is any pyCLAD vision model (e.g.
PatchCore above, RD4AD, FastFlow, PaSTe). It needs the concept id on every call, so it runs under
`ConceptAwareScenario` (`pyclad.scenarios.concept_aware`), not `ConceptIncrementalScenario`.

### Default profile

`ReplayCADConfig.for_benchmark(benchmark, artifact_dir, **overrides)` builds the paper's section
5.1 uniform profile: MVTec AD gets the LDM-256 profile below, VisA gets SD1.5-512, and any other
benchmark (BTech, DAGM, MPDD, multidataset streams) falls back to LDM-256 with a neutral
initializer, since the authors publish no configuration for those.

| | MVTec / LDM-256 | VisA / SD1.5-512 |
|---|---|---|
| model id | `CompVis/ldm-text2im-large-256` | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| resolution | 256 | 512 |
| conditioning dim `C` | 1280 | 768 |
| mask latent | 32×32×4 = 4096 | 64×64×4 = 16384 |
| MLP `(g, p)` | (128, 200) | (128, 192) |
| spatial tokens `M` (derived) | 5 | 32 |
| compression steps | 20 000 | 30 000 |
| compression batch size | 16 | 2 |
| MLP / embedding learning rate (derived) | 1.6e-3 / 1.6e-1 | 2.0e-4 / 2.0e-2 |

Shared by both profiles: `semantic_tokens=20`, AdamW `weight_decay=1e-2`,
`timestep_sampling="uniform"`, DDIM sampling (50 steps, `eta=0.0`), `guidance_scale=10.0`,
`replay_samples_per_concept=800`, up to 10 stored masks per concept.

`M` and the learning rates are derived, not typed in directly, and aren't guessable from the
config's field list alone:

- **`M`** = `(latent_values / mask_group_width) * mask_projection_width / condition_dim`.
  `ReplayCADConfig` validates that this divides evenly and rejects presets that don't — which is
  also why VisA's MLP width is `(128, 192)` here, not the paper's printed `(128, 196)` (not
  reshapeable: `128 * 196 / 768` isn't an integer).
- **Learning rates** come from three separate fields — `base_learning_rate` (`5e-5`),
  `learning_rate_scale` (the frozen `ngpu * batch_size` product) and `semantic_lr_multiplier`
  (`100.0`) — kept apart so the published rates don't silently rescale if you override
  `compression_batch_size` for your own hardware.

### Masks, augmentation and caching

ReplayCAD conditions generation on an object mask, but anomaly-detection datasets ship masks only
for anomalous *test* images — never for the normal training images compression learns from. The
authors ran Segment Anything over the training sets instead and publish the result as `SAM.zip`
for MVTec and VisA. For any other dataset you run SAM yourself: download a checkpoint from
[facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints)
(`sam_vit_h_4b8939.pth`, ~2.4 GB, is the `vit_h` variant the authors used; `vit_l` and `vit_b` are
smaller and also accepted) and point `sam_checkpoint` at it. Both `sam_checkpoint` and
`sam_model_type` are part of the compression cache key, so switching either invalidates every
artifact already computed — at 20000-30000 diffusion steps per concept, that is not a cheap
mistake.

`mask_backend` (default `"sam"`) selects `"sam"` (Segment Anything, needs `sam_checkpoint`),
`"precomputed"` (the authors' own masks, matched to training images by sorted filename order —
raises on a count mismatch), or `"full-frame"` (all-object mask, for texture concepts). Route
individual categories to a different backend with `mask_modes` — needed for MVTec's `zipper` and
VisA's `pipe_fryum`, which the authors' precomputed archive doesn't cover.

`mask_augmentation` (default `"none"`) jitters a stored mask at *replay* time — `"paper"`
(rotate + shift) or five class-specific transforms reproduced from the release's
`mask_transfor.py`; see `pyclad.vision.strategies.replaycad.masks` and `per_class.py` for which
class uses which.

Compressed artifacts (`embedding.pt`, `projection.pt`, up to `masks_per_concept` masks,
`meta.json`) are cached under `<artifact_dir>/<benchmark>/<concept_slug>/`, enabled by default.
The cache key (`meta.json`'s `config_hash`) covers everything that changes the compressed
representation and excludes concept ordering, the detector, and replay-only settings, so one
compression pass serves an entire benchmark sweep. `train_augmentation` (compression-time) is
hashed; `mask_augmentation` (replay-time) is not, since it jitters an already-compressed artifact
rather than changing it. A hash mismatch logs a warning and recompresses by default;
`strict_cache=True` raises instead.

### Per-class overrides

The authors' released scripts hand-tune every class instead of using the paper's uniform profile.
`apply_per_class(config, concept_id, benchmark)` (`pyclad.vision.strategies.replaycad.per_class`)
returns that alternative, re-validated profile for one concept, for inspection — it cannot drive a
live run end to end (divergence 2 below).

### Installation

```bash
pip install -e ".[replaycad]"
```

Installs `diffusers`, `transformers`, `accelerate`, `safetensors` and `segment-anything` on top of
the vision stack described in Setup above. They're imported lazily, so importing `pyclad` without
the extra keeps working; a missing import raises a message naming the extra.

### Differences from the original

This follows the authors' released hyperparameters but is not a bit-for-bit port. It runs on
`diffusers` instead of their vendored `ldm/`, trains a pyCLAD vision model instead of InvAD, and
defaults to the paper's uniform per-dataset profile rather than their per-class tuning — which
`per_class.py` records as a reference table. Several smaller behavioural differences exist in mask
handling and scoring. Check them before reporting a run here as a reproduction of the paper.

### Running the example

```bash
cd examples/strategies
python replaycad_example.py
```

Same pattern as the PatchCore example above — `read_vision_dataset`, the same metric callbacks,
`output.json` — plus a `ReplayCADConfig`/`ReplayCADMemory` built with `mask_backend="precomputed"`
against the authors' `SAM.zip` extracted to `./SAM/data` (pass `mask_backend="sam"` with a
`sam_checkpoint` instead if you don't have it) and `device="cuda"` (use `"cpu"` or `"mps"` if you
don't have one; compression is slow either way, so shrink `compression_steps` for a smoke test).
