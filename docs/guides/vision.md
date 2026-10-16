# Image anomaly detection

pyCLAD includes continual **visual anomaly detection**: image models that train on normal images and produce
per-pixel anomaly maps, readers for the common industrial benchmarks, and image- and pixel-level metrics. This guide
covers installing the extras, getting the data, and running a model. The models themselves, and how each one relates
to its paper, are described in [Vision](../vision.md).

Every model ships with a ready-to-run example under `examples/models/vision/`.

## Setup

The vision models are not part of the default pyCLAD install. Install them with the `vision` extra:

```bash
pip install "pyclad[vision]"
```

UCAD and ReplayCAD need a few more packages, which their own extras add on top of `vision`:
`pip install "pyclad[ucad]"` and `pip install "pyclad[replaycad]"`. From a clone of the repository, use
`pip install -e ".[vision]"` and so on.

| Package | Extra | Used for |
|---|---|---|
| `torch` | `torch` (included in `vision`) | networks, tensors, training |
| `pytorch-lightning` | `torch` (included in `vision`) | training loop (`pl.Trainer`, `LightningModule`) |
| `torchvision` | `vision` | pretrained backbones (ResNet / MobileNet / EfficientNet) + feature extraction |
| `pillow` | `vision` | image loading |
| `open_clip_torch` | `vision` | the Continual-MEGA baseline's CLIP backbone |
| `timm` | `ucad` | UCAD's ViT backbone |
| `segment-anything` | `ucad`, `replaycad` | SAM masks: UCAD's `structure_mode="sam"` and ReplayCAD's SAM mask backend |

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
--8<-- "docs/snippets/vision/read_dataset.py:read"
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

Computed on the per-image anomaly score, with the same base metrics as tabular data: `RocAuc`, `AveragePrecision`
and `F1Score` (see "Base metrics" in [Metrics](../metrics.md#base-metrics)).

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
--8<-- "docs/snippets/vision/pixel_callbacks.py:callbacks"
```

A metric can be undefined for a category: pixel-level metrics, for example, cannot be computed when a category's
test masks hold no anomalous pixel. By default a metric callback stops the run when that happens. The vision
example scripts pass `on_undefined="propagate"` to every metric callback instead, so a long run always finishes:
the category is logged and listed under `undefined_concepts` in the output, and the continual metrics that read
it are `NaN`. See "Undefined values" in [Metrics](../metrics.md#undefined-values).

## Multi-dataset streams

A continual stream can also draw its concepts from several source datasets, for example a few categories of MVTec AD
followed by some of VisA. A `MultiDatasetSpec` lists `DatasetBlock`s in stream order; each block names a source
dataset, the categories to take, and optionally its own root and a name prefix. Concepts are named
`<source>__<category>`, so categories with the same name in two datasets stay distinct.

A spec fixes the stream completely and round-trips through JSON, so save it with your results:

```python
--8<-- "docs/snippets/vision/multi_dataset_spec.py:spec"
```

`read_multi_dataset` reads the images, resolving each block against its own root or the `roots` mapping.
`load_inclad_md` loads the published InCLAD-MD benchmark, 27 categories from BTech, MPDD, DAGM, VisA and MVTec AD:

```python
--8<-- "docs/snippets/vision/multi_dataset_read.py:read"
```

A block can also be an explicit `FolderBenchmarkSpec` or `CsvBenchmarkSpec` for a dataset with a custom layout. See
`examples/multi_dataset_example.py`, and `examples/step_schedule_multi_dataset_example.py` for the same stream with a
step schedule.

