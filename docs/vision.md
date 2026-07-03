# Vision models — quick start

Continual **visual anomaly detection** models for pyCLAD. Each model trains on *normal* images only and produces per-pixel anomaly maps.

Every model ships with a ready-to-run example under `examples/models/vision/`.

## Models

| Model | What it is | Branch to check out | Example script |
|---|---|---|---|
| **PaSTe** | Student–teacher distillation | any vision branch (base) | `paste_torch_example.py` |
| **FastFlow** | Normalizing flow on features | `vision_fastflow` | `fastflow_torch_example.py` |

## Setup — extra libraries required

Beyond a normal pyCLAD install, the vision models need **exactly three additional packages** (the deep-learning stack):

| Package | Used for |
|---|---|
| `torch` | networks, tensors, training |
| `torchvision` | pretrained backbones (ResNet / MobileNet / EfficientNet) + feature extraction |
| `pytorch-lightning` | training loop (`pl.Trainer`, `LightningModule`) |

## Datasets

The example scripts read data from `examples/resources/vision/<dataset>/`. Datasets are *not* included in the repository, so after cloning you must download and place them yourself.

**1. Put the dataset here** (folder name is up to you; it just has to match the `root=` in the script):

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

python paste_torch_example.py      # PaSTe
python fastflow_torch_example.py   # FastFlow
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