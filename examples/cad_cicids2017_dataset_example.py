import logging
import pathlib

from pyod.models.vae import VAE

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.datasets.cad_cicids2017_dataset import CadCicids2017Dataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import PyODAdapter, IsolationForestAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer,
)
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

if __name__ == "__main__":
    """
    This example showcases how to run a concept aware scenario using the CAD-CICIDS2017 dataset adopted to
    continual anomaly detection. The dataset is automatically downloaded from Hugging Face on first use
    <https://huggingface.co/datasets/lifelonglab/CAD-CICIDS2017>.
    """
    dataset = CadCicids2017Dataset(ordering="curriculum_asc")
    model = IsolationForestAdapter()
    replay_buffer = AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=1000)
    strategy = ReplayEnhancedStrategy(model, replay_buffer)
    callbacks = [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        TimeEvaluationCallback(),
        MemoryUsageCallback(),
    ]
    scenario = ConceptAwareScenario(dataset, strategy=strategy, callbacks=callbacks)
    scenario.run()

    output_writer = JsonOutputWriter(pathlib.Path("output-cad-cicids2017.json"))
    output_writer.write([model, dataset, strategy, *callbacks])