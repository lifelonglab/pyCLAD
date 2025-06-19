import dataclasses
import logging
from typing import Any, Dict

from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from codecarbon.emissions_tracker import BaseEmissionsTracker
from codecarbon.output import EmissionsData, LoggerOutput

from pyclad.callbacks.callback import Callback
from pyclad.output.output_writer import InfoProvider


class InterceptorLogger(LoggerOutput):
    def __init__(self):
        self.emission_data = None
        logger = logging.getLogger(__name__)
        super().__init__(logger)

    def out(self, total: EmissionsData, delta: EmissionsData):
        self.emission_data = total
        super().out(total, delta)


class BaseEnergyCallback(Callback, InfoProvider):
    def __init__(self, tracker: BaseEmissionsTracker, logger: InterceptorLogger):
        self._tracker = tracker
        self._logger = logger
        self._train_co2 = 0
        self._eval_co2 = 0

    def before_training(self, *args, **kwargs):
        self._tracker.start()

    def after_training(self, *args, **kwargs):
        self._tracker.stop()

    def before_evaluation(self, *args, **kwargs):
        self._tracker.start()

    def after_evaluation(self, *args, **kwargs):
        self._tracker.stop()

    def info(self) -> Dict[str, Any]:
        return {"energy_evaluation_callback": {"emissions": dataclasses.asdict(self._logger.emission_data)}}


class EnergyEvaluationCallback(BaseEnergyCallback):
    def __init__(self):
        logger = InterceptorLogger()

        tracker = EmissionsTracker(logging_logger=logger, save_to_logger=True, save_to_file=False)
        super().__init__(tracker, logger)


class OfflineEnergyEvaluationCallback(BaseEnergyCallback):
    def __init__(self, country_iso_code, region: None, cloud_provider: None, cloud_region: None):
        logger = InterceptorLogger()

        tracker = OfflineEmissionsTracker(
            country_iso_code=country_iso_code,
            region=region,
            cloud_provider=cloud_provider,
            cloud_region=cloud_region,
            logging_logger=logger,
            save_to_logger=True,
            save_to_file=False,
        )
        super().__init__(tracker, logger)
