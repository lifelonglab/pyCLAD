from typing import Callable, Optional

import numpy as np
import open_clip
import torch

from pyclad.vision.models.continual_mega_baseline.backbone import build_clip_backbone
from pyclad.vision.models.continual_mega_baseline.config import (
    ContinualMegaBaselineConfig,
)
from pyclad.vision.models.continual_mega_baseline.prompt_learner import PromptLearner
from pyclad.vision.models.continual_mega_baseline.prompts import (
    ANOMALY_PROMPTS,
    NORMAL_PROMPTS,
)
from pyclad.vision.models.continual_mega_baseline.scoring import anomaly_scores_and_maps
from pyclad.vision.models.continual_mega_baseline.text_encoder import (
    ClipTextEncoder,
    encode_prompt_groups,
)
from pyclad.vision.models.continual_mega_baseline.trainer import (
    ContinualMegaBaselineTrainer,
)
from pyclad.vision.models.continual_mega_baseline.visual_encoder import (
    AdaptedVisualEncoder,
)
from pyclad.vision.models.supervised_vision_model import SupervisedVisionModel
from pyclad.vision.prediction_results import VisionPredictionResults

PROMPT_GROUPS = {"normal": NORMAL_PROMPTS, "abnormal": ANOMALY_PROMPTS}
IMAGE_VALUE_RANGE = 255.0


class ContinualMegaBaseline(SupervisedVisionModel):
    def __init__(self, config: ContinualMegaBaselineConfig):
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        clip_model = build_clip_backbone(config.clip_model_name, config.weights_path)
        clip_model.requires_grad_(False)
        clip_model.eval()
        self.visual_encoder = AdaptedVisualEncoder(
            visual=clip_model.visual,
            feature_layers=config.feature_layers,
            width=config.width,
            bottleneck=config.bottleneck,
            adapter_weight=config.adapter_weight,
            noise_sigma=config.noise_sigma,
            noise_in_float32=config.noise_in_float32,
        ).to(self.device)
        self.prompt_learner = PromptLearner(
            token_embedding=clip_model.token_embedding,
            context_dim=clip_model.ln_final.weight.shape[0],
            prompt_groups=PROMPT_GROUPS,
            n_ctx=config.n_ctx,
            tokenize=open_clip.tokenize,
        ).to(self.device)
        self.text_encoder = ClipTextEncoder(clip_model).to(self.device)
        self.visual_encoder.eval()
        self.prompt_learner.eval()
        self.text_encoder.eval()
        self.trainer = ContinualMegaBaselineTrainer(
            visual_encoder=self.visual_encoder,
            prompt_learner=self.prompt_learner,
            text_features=self.text_features,
            config=config,
            train_prompts=config.train_prompts,
        )

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        masks: Optional[np.ndarray] = None,
        on_epoch_end: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        images = self._to_tensor(data)
        targets = torch.from_numpy(labels)
        pixel_targets = self._mask_tensor(masks, labels)
        while self.trainer.epochs_completed < self.config.epochs:
            loss = self.trainer.train_epoch(images, targets, pixel_targets)
            if on_epoch_end is not None:
                on_epoch_end(self.trainer.epochs_completed, loss)
        self.trainer.epochs_completed = 0

    def text_features(self) -> torch.Tensor:
        return encode_prompt_groups(self.text_encoder, self.prompt_learner())

    def predict(self, data: np.ndarray) -> VisionPredictionResults:
        scores, maps = [], []
        with torch.no_grad():
            text_features = self.text_features()
            for start in range(0, len(data), self.config.batch_size):
                batch = self._to_tensor(data[start : start + self.config.batch_size])
                tokens, _ = self.visual_encoder(batch)
                batch_scores, batch_maps = anomaly_scores_and_maps(
                    patch_tokens=tokens,
                    text_features=text_features,
                    logit_scale=self.config.logit_scale,
                    output_size=self.config.image_size,
                )
                scores.append(batch_scores.cpu().numpy())
                maps.append(batch_maps.cpu().numpy())

        anomaly_scores = np.concatenate(scores).astype(np.float32)
        return VisionPredictionResults(
            y_pred=(anomaly_scores > 0.5).astype(np.int64),
            anomaly_scores=anomaly_scores,
            score_maps=np.concatenate(maps).astype(np.float32),
        )

    def _mask_tensor(self, masks: Optional[np.ndarray], labels: np.ndarray) -> torch.Tensor:
        if masks is None:
            if labels.any():
                raise ValueError(f"{self.name()} needs ground truth masks to train on anomalous samples")
            return torch.zeros((len(labels), self.config.image_size, self.config.image_size))
        return torch.from_numpy(masks).float()

    def _to_tensor(self, images: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        return tensor.permute(0, 3, 1, 2).float().div_(IMAGE_VALUE_RANGE)

    def name(self) -> str:
        return "Continual-MEGA baseline"
