# Vision models — quick start

Continual **visual anomaly detection** models for pyCLAD. Each model trains on *normal* images only and produces per-pixel anomaly maps.

Every model ships with a ready-to-run example under `examples/models/vision/`.

## Models

| Model | What it is | Example script |
|---|---|--|
| **PaSTe** | Student–teacher distillation | `paste_torch_example.py` |
| **FastFlow** | Normalizing flow on features | `fastflow_torch_example.py` |
| **UCAD** | Continual key-prompt-knowledge memory over a frozen ViT | `ucad_torch_example.py` |

### UCAD

Liu et al., *Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt*,
AAAI 2024 ([reference code](https://github.com/shirowalker/UCAD)). Unlike the other two
models, UCAD is itself the continual mechanism: each concept appends one entry to a memory
that is never reset — a **key** coreset of frozen-ViT patch features, a **prompt** of learned
attention prefixes, and a **knowledge** coreset scored against by nearest neighbour. At
inference each image is routed to a task by its nearest key, so task identity is inferred
from the data and never supplied by the caller. Use `UCADStrategy`, not a replay wrapper, and
a `ConceptAwareScenario` — `learn()` needs the concept name to find its SAM structure masks,
while `predict()` discards it.

`structure_mode` defaults to `"none"`, which skips the contrastive prompt training (the
paper's CPM-only ablation) and needs no extra data. For the full method, point
`structure_mask_root` at the authors' precomputed SAM maps, laid out as
`<root>/<category>/train/good/*.png`.

#### Data leakage

**In this port: none.** `fit()` accepts only the training array, the decision threshold comes
from a quantile of training scores, and routing reads only the learned keys.
`tests/vision/models/test_ucad_no_leakage.py` pins that as executable assertions — scores must
not change when the evaluated batch is regrouped or reordered, `predict()` must not mutate
learned state, and the strategy must ignore the concept id it is handed.

Which is why these numbers are *not* comparable to the paper's Tables 1–4, and are expected to be lower. In `run_ucad.py`, every
epoch is evaluated on the test set and the prompt and knowledge bank are kept from the epoch
with the best test AUROC (lines 262–264, with an early break at AUROC = 1); the reported score
is a running ensemble accumulated across epochs (line 128 vs 149); scores are min-max
normalised over the whole test set (lines 190–211); the continual evaluation loop is commented
out entirely (lines 410–509), so the published per-task numbers are measured right after each
task is trained and cannot show forgetting; and the reported "FM" is the gap between two memory
budgets, not forgetting in the sense of Chaudhry et al. There is no validation split
(`train_val_split=1`), so honest epoch selection is impossible in that codebase — this port
therefore trains a fixed 25 epochs and keeps the last one, with no selection criterion at all.

## Setup — extra libraries required

Beyond a normal pyCLAD install, the vision models need the deep-learning stack below. PaSTe and
FastFlow need only the first three packages; UCAD additionally needs `timm`, and
`segment-anything` if you use `structure_mode="sam"`. Install everything UCAD needs at once
with `pip install -e '.[ucad]'`.

| Package | Used for |
|---|---|
| `torch` | networks, tensors, training |
| `torchvision` | pretrained backbones (ResNet / MobileNet / EfficientNet) + feature extraction |
| `pytorch-lightning` | training loop (`pl.Trainer`, `LightningModule`) |
| `timm` | UCAD's ViT backbone |
| `segment-anything` | UCAD's `structure_mode="sam"` only |

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