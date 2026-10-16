# Installation

pyCLAD needs Python 3.10 or newer and is available as a [package on PyPI](https://pypi.org/project/pyclad/):

```
pip install pyclad
```

## Optional dependencies

The default installation covers the scenarios, metrics, callbacks, datasets, the baseline and replay strategies, and
the PyOD-based models. Heavier dependencies are kept out of it and installed through extras:

| Extra | Install with | Adds |
|---|---|---|
| `torch` | `pip install "pyclad[torch]"` | PyTorch models (autoencoders, CARLA) and the strategies that train them (EWC, LwF, DER++, A-GEM) |
| `vision` | `pip install "pyclad[vision]"` | image anomaly detection models, datasets and pixel-level metrics (includes `torch`) |
| `ucad` | `pip install "pyclad[ucad]"` | the UCAD model and strategy (includes `vision`) |
| `replaycad` | `pip install "pyclad[replaycad]"` | the ReplayCAD strategy (includes `vision`) |
| `all` | `pip install "pyclad[all]"` | everything above |

Without a GPU, installing the CPU build of PyTorch first avoids downloading several GB of CUDA libraries:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "pyclad[torch]"
```

pyCLAD can use any model from the [PyOD library](https://pyod.readthedocs.io/en/latest/); some PyOD models need
additional packages of their own.

## Installing for development

To change pyCLAD itself, clone the repository and install it in editable mode with the development tools:

```
git clone https://github.com/lifelonglab/pyCLAD.git
cd pyCLAD
pip install -e ".[dev]"
pytest tests
```

See [Contributing](../contribution.md) for the rest of the workflow.
