import logging
import pathlib

from pyclad.callbacks.evaluation.grouped_concept_metric_evaluation import (
    GroupedConceptMetricCallback,
)
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.final_step_average import FinalStepAverage
from pyclad.metrics.continual.forgetting_measure_strict import ForgettingMeasureStrict
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.supervised_concept_incremental import (
    SupervisedConceptIncrementalScenario,
)
from pyclad.vision.callbacks.grouped_vision_pixel_concept_metric_callback import (
    GroupedVisionPixelConceptMetricCallback,
)
from pyclad.vision.data.benchmarks.continual_mega import ContinualMegaBenchmarkReader
from pyclad.vision.metrics.pixel_average_precision import PixelAveragePrecision
from pyclad.vision.models.continual_mega_baseline.config import (
    ContinualMegaBaselineConfig,
)
from pyclad.vision.models.continual_mega_baseline.continual_mega_baseline import (
    ContinualMegaBaseline,
)
from pyclad.vision.strategies.naive_supervised import NaiveSupervisedStrategy

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    reader = ContinualMegaBenchmarkReader(
        data_root=pathlib.Path("../../resources/vision/continual_mega"),
        meta_dir=pathlib.Path("../../resources/vision/continual_mega/meta_files"),
        scenario=2,
        task_size=30,
        zero_shot=True,
        train_samples="all",
    )
    dataset = reader.read_dataset()

    model = ContinualMegaBaseline(
        ContinualMegaBaselineConfig(
            weights_path=pathlib.Path("../../resources/vision/clip/ViT-L-14-336px.pt"),
            epochs=50,
            train_batch_size=16,
            learning_rate=1e-4,
        )
    )
    strategy = NaiveSupervisedStrategy(model)

    groups = dataset.group_by_concept()
    summarized_metrics = [FinalStepAverage(), ForgettingMeasureStrict()]
    callbacks = [
        GroupedConceptMetricCallback(RocAuc(), groups, summarized_metrics),
        GroupedVisionPixelConceptMetricCallback(PixelAveragePrecision(), groups, summarized_metrics),
        TimeEvaluationCallback(),
    ]

    SupervisedConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=callbacks).run()

    JsonOutputWriter(pathlib.Path("output.json")).write([model, dataset, strategy, *callbacks])
