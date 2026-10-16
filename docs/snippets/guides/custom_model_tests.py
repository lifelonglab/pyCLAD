# --8<-- [start:test]
import numpy as np
import pytest

from pyclad.callbacks import ConceptMetricCallback
from pyclad.data import Concept, ConceptsDataset
from pyclad.metrics import ContinualAverage, RocAuc
from pyclad.scenarios import ConceptIncrementalScenario
from pyclad.strategies.baselines.naive import NaiveStrategy

# from mypackage.models import SklearnIsolationForest  (the model from "Writing your own model")


@pytest.fixture(scope="module")
def results():
    """Run the model through a scenario once; the tests below assert on the result."""
    rng = np.random.default_rng(0)
    train = [Concept(f"c{i}", data=rng.normal(i * 5, 1, size=(200, 3))) for i in range(2)]
    test = [
        Concept(
            f"c{i}",
            data=np.concatenate([rng.normal(i * 5, 1, (45, 3)), rng.normal(i * 5 + 4, 1, (5, 3))]),
            labels=np.array([0] * 45 + [1] * 5),
        )
        for i in range(2)
    ]
    callback = ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[ContinualAverage()])
    dataset = ConceptsDataset("Synthetic", train_concepts=train, test_concepts=test)
    model = SklearnIsolationForest(random_state=0)
    ConceptIncrementalScenario(dataset, strategy=NaiveStrategy(model), callbacks=[callback]).run()
    return callback.info()["concept_metric_callback_ROC-AUC"]


def test_detects_anomalies_of_the_concept_it_just_learned(results):
    # Would fail if the score's sign were flipped: ROC-AUC would be near 0, not near 1.
    for concept in ["c0", "c1"]:
        assert results["metric_matrix"][concept][concept] > 0.9


def test_reports_its_settings():
    assert SklearnIsolationForest(random_state=3).info() == {
        "model": {"name": "SklearnIsolationForest", "n_estimators": 100, "random_state": 3}
    }


# --8<-- [end:test]


if __name__ == "__main__":
    import runpy
    from pathlib import Path

    # In the repository the model would be imported; here it comes from the guide's model snippet.
    SklearnIsolationForest = runpy.run_path(str(Path(__file__).parents[1] / "models" / "custom_model.py"))[
        "SklearnIsolationForest"
    ]
    test_detects_anomalies_of_the_concept_it_just_learned(results.__wrapped__())
    test_reports_its_settings()
