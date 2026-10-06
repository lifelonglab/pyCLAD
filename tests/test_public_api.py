import importlib
import subprocess
import sys

import pytest

PACKAGES = [
    "pyclad.callbacks",
    "pyclad.data",
    "pyclad.data.datasets",
    "pyclad.data.readers",
    "pyclad.metrics",
    "pyclad.scenarios",
]


@pytest.mark.parametrize("package_name", PACKAGES)
def test_every_exported_name_is_importable(package_name):
    package = importlib.import_module(package_name)

    assert package.__all__ == sorted(set(package.__all__))
    for name in package.__all__:
        assert hasattr(package, name), f"{package_name}.__all__ lists {name!r}, which the package does not define"


def test_short_imports_name_the_same_objects_as_the_full_paths():
    from pyclad.callbacks import ConceptMetricCallback
    from pyclad.callbacks.evaluation.concept_metric_evaluation import (
        ConceptMetricCallback as FullPathCallback,
    )
    from pyclad.data import ConceptsDataset
    from pyclad.data.datasets import ConceptsDataset as DatasetsConceptsDataset
    from pyclad.data.datasets.concepts_dataset import (
        ConceptsDataset as FullPathConceptsDataset,
    )

    assert ConceptMetricCallback is FullPathCallback
    assert ConceptsDataset is DatasetsConceptsDataset is FullPathConceptsDataset


def test_importing_the_core_packages_does_not_load_optional_or_slow_dependencies():
    # A fresh interpreter, because this test session has already imported torch and datasets.
    code = (
        "import sys\n"
        + "".join(f"import {package}\n" for package in PACKAGES)
        + "print(','.join(name for name in ('torch', 'pytorch_lightning', 'datasets') if name in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == ""
