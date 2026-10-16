# Vision

This page describes the image anomaly detection models and strategies in `pyclad.vision`, and how each one relates
to its paper. To install the extras, get the data and run a first model, see
[Image anomaly detection](guides/vision.md).

## Models

| Model | What it is | Example script |
|---|---|---|
| **PatchCore** | Memory bank of pretrained patch features, nearest-neighbour scoring | `patchcore_torch_example.py` |
| **FastFlow** | Normalizing flow on features | `fastflow_torch_example.py` |
| **RD4AD** | Reverse distillation from a pretrained encoder | `rd4ad_torch_example.py` |
| **PaSTe** | Student–teacher distillation | `paste_torch_example.py` |
| **UCAD** | Continual key-prompt-knowledge memory over a frozen ViT | `ucad_torch_example.py` |
| **Continual-MEGA baseline** | CLIP with MoE adapters and learned prompts, supervised | `continual_mega_example.py` |

All examples are in `examples/models/vision/`. Image models plug into the usual strategies (for example
`ReplayEnhancedStrategy`); UCAD and ReplayCAD below are continual strategies of their own.

## UCAD

Liu et al., *Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt*,
AAAI 2024 ([reference code](https://github.com/shirowalker/UCAD)). Unlike the other
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

### Data leakage

**In this port: none.** `fit()` accepts only the training array, the decision threshold comes
from a quantile of training scores, and routing reads only the learned keys.
`tests/vision/models/test_ucad_no_leakage.py` pins that as executable assertions — scores must
not change when the evaluated batch is regrouped or reordered, `predict()` must not mutate
learned state, and the strategy must ignore the concept id it is handed.

Results from this port are not comparable to the paper's Tables 1–4 and are expected to be lower.
The reference code selects the epoch and post-processes scores using the test set, and reports
per-task scores right after each task is trained, so its numbers cannot show forgetting. pyCLAD's
protocol allows neither. The reference has no validation split to select epochs on honestly, so
this port trains a fixed 25 epochs and keeps the last one.

??? note "Details: the reference evaluation, line by line"
    In the reference code's `run_ucad.py`:

    - every epoch is evaluated on the test set, and the prompt and knowledge bank are kept from the
      epoch with the best test AUROC (lines 262–264, with an early break at AUROC = 1);
    - the reported score is a running ensemble accumulated across epochs (line 128 vs 149);
    - scores are min-max normalised over the whole test set (lines 190–211);
    - the continual evaluation loop is commented out (lines 410–509), so each task is evaluated
      only right after it is trained;
    - the reported "FM" compares two memory budgets; it is not forgetting in the sense of
      Chaudhry et al.;
    - there is no validation split (`train_val_split=1`).

### Divergences from the reference

`UCADConfig`'s defaults follow the [reference code](https://github.com/shirowalker/UCAD) rather
than the paper prose wherever the two disagree. Every place this port departs from either is
listed here.

| What | Reference / paper | This port | Why |
|---|---|---|---|
| Epoch selection | Keeps the epoch with the best **test** AUROC (`run_ucad.py` lines 262–264) | Trains a fixed `epochs=25` and keeps the last | No test data may reach `fit()` — see [Data leakage](#data-leakage) above |
| Score post-processing | Min-max normalises over the whole test set and ensembles scores across epochs | Raw nearest-neighbour distances from the final epoch | Same reason |
| Backbone weights | The paper's text says ImageNet-21k; the code's `vit_base_patch16_224` + `pretrained=True` resolved (timm 0.6.7) to ImageNet-21k **finetuned on ImageNet-1k** | `backbone_name="vit_base_patch16_224.augreg_in21k_ft_in1k"` — the weights the reference actually loaded | Newer timm requires the tag to be explicit; pinning it keeps the port reproducible. Set `"vit_base_patch16_224.augreg_in21k"` for the paper's stated variant |
| `prompt_depth` | Prompts all 12 blocks while reading features out after block index 5 | Defaults to `6`, matching `feature_layer` | Prompt slices past the read-out block receive no gradient and only cost storage. Larger values are still accepted and merely log an info message |
| Coreset size | Samples a *percentage* of the feature pool | Samples an exact count (`key_size` / `knowledge_size`, both `196`) | A percentage of a varying pool can round down by one; UCAD specifies the banks as absolute sizes |

Reproduced deliberately, quirks included: the SCL loss keeps the reference's diagonal self-pairs
(a constant offset that does not change the gradient direction); patch-label downsampling
reproduces `cv2.resize`'s bilinear-then-round behaviour (`mask_interpolation="bilinear"`);
routing distances are squared to match the reference's faiss index; and the prompt is
initialised `uniform(-1, 1)` as in EPrompt.

## PatchCore

PatchCore is a memory-bank detector: a frozen, pretrained backbone extracts mid-level patch
features (`layer2` + `layer3`) for every training image, a greedy coreset subsamples them into a
small memory bank, and each test patch is scored by nearest-neighbour distance to that bank. No
gradient training happens — `fit()` only extracts features and builds the memory bank.

The image-level score is the maximum over **raw** patch distances; the segmentation map is those
same distances upsampled and smoothed (divergence 6 below).

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

`patchcore_torch_example.py` runs it like the other image examples; see "Run a model" in
[Image anomaly detection](guides/vision.md#run-a-model).

## ReplayCAD

ReplayCAD (Hu et al., *ReplayCAD: Generative Diffusion Replay for Continual Anomaly Detection*,
IJCAI 2025, arXiv:2505.06603) is a generative-replay strategy: instead of storing real images from
earlier concepts, it compresses each one into a small, frozen-diffusion-model conditioning (a
learned semantic embedding plus a mask-projection MLP) that can regenerate representative images
on demand, and replays those alongside new data when refitting the detector.

The port follows the authors' code at [HULEI7/ReplayCAD](https://github.com/HULEI7/ReplayCAD),
commit `acc6195`.

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
the vision stack described in "Setup" in [Image anomaly detection](guides/vision.md#setup). They're imported lazily, so importing `pyclad` without
the extra keeps working; a missing import raises a message naming the extra.

### Running the example

```bash
cd examples/strategies
python replaycad_example.py
```

Same pattern as the other image examples, plus a `ReplayCADConfig`/`ReplayCADMemory` built with `mask_backend="precomputed"`
against the authors' `SAM.zip` extracted to `./SAM/data` (pass `mask_backend="sam"` with a
`sam_checkpoint` instead if you don't have it) and `device="cuda"` (use `"cpu"` or `"mps"` if you
don't have one; compression is slow either way, so shrink `compression_steps` for a smoke test).

## Differences from the original

Neither model here is a bit-for-bit port. PatchCore matches the reference implementation behind
ReplayCAD's benchmark row (ADer's `model/patchcore.py`) constant-for-constant; ReplayCAD follows
the authors' released hyperparameters. Where either departs from its source, it is listed below.
Check this list before reporting a run here as a reproduction of the paper.

<!-- Docstrings and the sections above refer to these entries by number ("divergence 6 below"); renumbering them
     means updating those references. -->

1. **Detector.** `ReplayCADStrategy` refits any pyCLAD vision model — PatchCore above, RD4AD,
   FastFlow, PaSTe — while the authors train InvAD. Replay quality is therefore measured through a
   different detector than the paper's tables, so absolute numbers are not comparable even where
   the replay itself matches.

2. **Per-class tuning is a reference table, not a runnable mode.** The default is the paper's
   section 5.1 uniform per-dataset profile, built by `ReplayCADConfig.for_benchmark`.
   `apply_per_class` returns the authors' released hand-tuned settings for one concept, for
   inspection only: it varies `model_id`, `condition_dim` and `resolution` per class, while
   `ReplayCADMemory` and `DiffusersBackend` each hold one config for the whole stream. A live run
   driven through it would fail on a shape mismatch — VisA alone mixes the LDM-256 and SD1.5-512
   families.

3. **Diffusion stack.** Compression and generation run on `diffusers`, not the release's vendored
   `ldm/` tree, and the learned semantic embedding is injected through a forward hook on the text
   encoder's embedded output rather than trained in place.

4. **VisA mask-projection width** is `(128, 192)`, not the paper's printed `(128, 196)`. The
   derived token count `M` must divide evenly, and `128 * 196 / 768` is not an integer;
   `ReplayCADConfig` rejects presets that cannot be reshaped rather than silently truncating.

5. **Benchmarks without a published profile.** The authors publish configurations for MVTec and
   VisA only. BTech, DAGM, MPDD and multidataset streams fall back to the LDM-256 preset with a
   neutral initializer, and `for_benchmark` logs a warning when that happens. Those runs reproduce
   nothing published — they are the method applied to a new dataset.

6. **PatchCore's image-level score is taken before smoothing.** It is the maximum over raw patch
   distances, while the returned segmentation map is those same distances bilinearly upsampled and
   Gaussian-smoothed with `smoothing_sigma`. The image score is therefore generally larger than the
   maximum of the map — which matters when comparing against ADer's published numbers.

7. **`random_reset` drops a component rather than misplacing it.** Both implementations give each
   connected component 100 attempts to find a non-overlapping position. When one exhausts them, the
   release records no position for it and then pastes through `zip(objects, positions)`, which
   shifts every later component onto another component's position and drops the last one; this port
   drops only the component that failed. No per-class preset uses this transform; it applies only if you set
   `mask_augmentation="random_reset"` yourself.

8. **Connected components** are found with `scipy.ndimage.label` (4-connectivity) instead of the
   release's `cv2.findContours(RETR_EXTERNAL)` (8-connectivity, filled contours), in the transform
   mirroring `visa_candle`. Equivalent for the hole-free, non-diagonally-touching components the
   affected VisA masks actually have.

9. **Where the released scripts disagree with themselves, this port picks one value.** Three cases:
   generation checkpoints past their training config's `max_steps`, an `--init_word` phrase
   truncated to its first word by a single-value argparse option, and two dead, mutually
   contradictory `macaroni2` branches. The module docstring of
   `pyclad.vision.strategies.replaycad.per_class` says which value this port takes, and why.

## Continual-MEGA

A continual anomaly-detection benchmark spanning seven datasets. Unlike the readers above it defines a
whole protocol — which classes form the base task, how the rest are split into tasks, which datasets are
held out for zero-shot — so it has its own reader instead of a `benchmark=` name.

**1. Put the data here**, under the folder names the benchmark metadata expects:

```
examples/resources/vision/continual_mega/
├── continual_ad/
├── mvtec_anomaly_detection/
├── VisA_20220922/
├── VIADUCT/
├── Real-IAD-512/
├── MPDD/
├── BTAD/
└── meta_files/              # copied from the reference repository
```

**2. Where to get it**: ContinualAD and the `meta_files/` directory come from the benchmark
[dataset page](https://huggingface.co/datasets/Continual-Mega/Continual-MEGA-Benchmark) and
[reference repository](https://github.com/Continual-Mega/Continual-MEGA-Baseline); MVTec-AD, VisA, MPDD
and BTAD are listed under "Datasets" in [Image anomaly detection](guides/vision.md#datasets); Real-IAD and VIADUCT come from their own sources.

**3. Pick a scenario:**

| Scenario | Base classes | New classes | Zero-shot |
|---|---|---|---|
| 1 | 85 (all datasets) | 60, split into 12 / 6 / 2 tasks | – |
| 2 | 58 (no MVTec-AD, no VisA) | 60, split into 12 / 6 / 2 tasks | MVTec-AD, VisA |
| 3 | 58 (also no ContinualAD) | 30, split into 6 / 3 / 1 tasks | MVTec-AD, VisA |

```python
--8<-- "docs/snippets/vision/continual_mega.py:reader"
```

Training concepts are the task groups (`base`, `task_1`, …) while test concepts are individual classes.

### Grouped metrics

One training concept covers many test concepts here, so the benchmark reports a score per task group
rather than per class. The grouped callbacks average each group's classes into a single cell before the
summarized metrics see the matrix; averaging over all classes instead would weight the larger groups
more and give different numbers:

```python
--8<-- "docs/snippets/vision/continual_mega.py:grouped"
```

Held-out zero-shot groups are reported separately under `held_out_groups` and excluded from both metrics.

For a per-class view, add the plain `ConceptMetricCallback` alongside them: its matrix has one column per
class and one row per task group, which is the rectangular shape the
[schedule-aware metrics](metrics.md#rectangular-matrices-and-step-schedules) describe. `dataset.first_seen_step()` supplies the mapping they need
and answers what the grouped metrics cannot — whether a class was learned at all, or was already that
good before training:

```python
--8<-- "docs/snippets/vision/continual_mega.py:schedule_aware"
```

`first_seen_step()` requires `zero_shot=False`: held-out classes never enter training, so no training step
describes them.

### Supervised models

`train_samples="all"` gives 10 normal and 10 anomalous images per class with pixel masks, so the model
trains on supervision rather than on normal data alone. Vision models of this kind implement
`SupervisedVisionModel`, which adds an optional `masks` argument to
[`SupervisedModel.fit`](models.md); `NaiveSupervisedStrategy` passes a concept's masks through to it, and
the stream runs under [`SupervisedConceptIncrementalScenario`](scenarios.md):

```python
from pyclad.vision.strategies.naive_supervised import NaiveSupervisedStrategy

strategy = NaiveSupervisedStrategy(model)
```

`ContinualMegaBaseline` (`continual_mega_example.py`) is the benchmark's own model: CLIP with adapters
and learned prompts. Set `train_samples="normal"` instead to run one-class models such as PaSTe or
FastFlow on the same streams with the usual strategy and scenario.
