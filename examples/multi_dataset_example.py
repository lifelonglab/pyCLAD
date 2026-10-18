import logging
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.average_precision import AveragePrecision
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.benchmarks.hf_benchmark import (
    load_inclad_bench,
    split_multi_dataset_concept_name,
)
from pyclad.vision.metrics.pixel_aupro import PixelAUPRO
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.rd4ad.config import RD4ADConfig
from pyclad.vision.models.rd4ad.rd4ad import RD4AD

logging.basicConfig(level=logging.INFO)

RESOURCES = pathlib.Path(__file__).resolve().parent / "resources" / "vision"
ROOTS = {
    "btech": RESOURCES / "BTech_Dataset_transformed",
    "dagm": RESOURCES / "DAGM_KaggleUpload",
    "mpdd": RESOURCES / "MPDD",
    "mvtec": RESOURCES / "mvtec_ad",
    "visa": RESOURCES / "VisA",
}

ORDERING = "easy_to_hard"

if __name__ == "__main__":
    """
    This example runs the `inclad-md` config of the InCLAD benchmark: a single continual
    stream of 27 categories drawn from five source datasets at once -- BTech, MPDD, DAGM,
    VisA and MVTec AD. The sample lists and the category order come from Hugging Face; the
    images are read from the local folders in ROOTS, each source from its own folder.

    Concepts are named `<source>__<category>`, so categories that share a name across
    datasets stay distinct. See "InCLAD benchmark" in docs/vision.md.
    """
    dataset = load_inclad_bench(
        "inclad-md",
        roots=ROOTS,
        ordering=ORDERING,
        resize_to=(256, 256),
        max_train_samples_per_category=100,
    )

    print("Stream:", [concept.name for concept in dataset.train_concepts()])
    per_source: dict[str, int] = {}
    for concept in dataset.train_concepts():
        source, _ = split_multi_dataset_concept_name(concept.name)
        per_source[source] = per_source.get(source, 0) + 1
    print("Concepts per source dataset:", per_source)

    model = RD4AD(
        RD4ADConfig(
            input_size=(256, 256),
            backbone_name="resnet18",
            pretrained_encoder=True,
            freeze_encoder=True,
            batch_size=16,
            epochs=2,
            learning_rate=5e-3,
            score_smoothing_sigma=4.0,
            score_mode="max",
            threshold_quantile=0.99,
            show_training_progress=True,
        )
    )
    strategy = NaiveStrategy(model)
    summarized_metrics = [ContinualAverage(), BackwardTransfer(), ForwardTransfer()]

    callbacks = [
        # Image-level
        ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=summarized_metrics, on_undefined="propagate"),
        ConceptMetricCallback(
            base_metric=AveragePrecision(), summarized_metrics=summarized_metrics, on_undefined="propagate"
        ),
        # Pixel-level
        VisionPixelConceptMetricCallback(
            base_metric=PixelRocAuc(), summarized_metrics=summarized_metrics, on_undefined="propagate"
        ),
        VisionPixelConceptMetricCallback(
            base_metric=PixelAUPRO(), summarized_metrics=summarized_metrics, on_undefined="propagate"
        ),
        TimeEvaluationCallback(),
    ]

    scenario = ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path(f"output_inclad_md_{ORDERING}.json"))
    output_writer.write([model, dataset, strategy, *callbacks])
