from __future__ import annotations

import inspect
import logging
from typing import Optional, Protocol

import numpy as np
import torch

from pyclad.vision.strategies.replaycad.artifacts import ConceptArtifact
from pyclad.vision.strategies.replaycad.config import ReplayCADConfig
from pyclad.vision.strategies.replaycad.masks import augment_mask, augment_training_pair
from pyclad.vision.strategies.replaycad.spatial import MaskProjection

logger = logging.getLogger(__name__)


class DiffusionBackend(Protocol):
    def compress(self, concept_id: str, images: np.ndarray, masks: np.ndarray) -> tuple[torch.Tensor, Optional[dict]]:
        """Learn the concept's semantic embedding and mask projection."""

    def generate(self, artifact: ConceptArtifact, count: int, seed: int) -> np.ndarray:
        """Generate ``count`` normal images for the concept described by ``artifact``."""

    def release_device_memory(self) -> None:
        """Move accelerator state off the device between ReplayCAD phases."""


class DiffusersBackend:
    """ReplayCAD stage 1 and 2 on frozen Diffusers components.

    Only the semantic placeholder vectors and the mask projection are trained; VAE, text encoder
    and U-Net (including its embedding matrix) stay frozen. Placeholders are substituted into the
    text encoder's embedded output via a forward hook rather than trained in place, mirroring the
    released ``EmbeddingManager`` -- which is what makes a shared, nonzero ``weight_decay`` safe:
    it never touches the untrained vocabulary rows.
    """

    def __init__(self, config: ReplayCADConfig):
        self.config = config
        self._pipeline = None
        self._placeholder_ids: Optional[list[int]] = None

    def _load(self):
        if self._pipeline is not None:
            self._pipeline.to(self.config.device)
            return self._pipeline

        try:
            from diffusers import DDIMScheduler, DiffusionPipeline
        except ImportError as exc:
            raise ImportError(
                "ReplayCAD's diffusion backend requires diffusers and transformers. "
                "Install pyclad with the 'replaycad' extra: pip install -e '.[replaycad]'"
            ) from exc

        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            self.config.torch_dtype
        ]
        pipeline = DiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            local_files_only=self.config.local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.to(self.config.device)

        for module in (self._text_encoder(pipeline), pipeline.unet, self._vae(pipeline)):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        cross_attention_dim = int(getattr(pipeline.unet.config, "cross_attention_dim", self.config.condition_dim))
        if cross_attention_dim != self.config.condition_dim:
            raise ValueError(
                f"ReplayCAD preset expects condition_dim={self.config.condition_dim}, but "
                f"'{self.config.model_id}' exposes {cross_attention_dim}"
            )

        self._pipeline = pipeline
        return pipeline

    def _text_encoder(self, pipeline):
        return getattr(pipeline, "text_encoder", None) or pipeline.bert

    def _vae(self, pipeline):
        return getattr(pipeline, "vae", None) or pipeline.vqvae

    def release_device_memory(self) -> None:
        if self._pipeline is None or not self.config.device.startswith("cuda"):
            return
        self._pipeline.to("cpu")
        torch.cuda.empty_cache()
        logger.info("ReplayCAD diffusion pipeline moved to CPU")

    def _install_placeholders(self, pipeline, concept_id: str) -> list[int]:
        """Register K placeholder tokens and initialize their embeddings.

        The authors repeat one initializer word's embedding K times (``embedding_manager.py:80``);
        the paper instead says the embedding is randomly initialized. ``initializer_word=None``
        (the default) follows the paper.
        """
        tokenizer = pipeline.tokenizer
        text_encoder = self._text_encoder(pipeline)

        tokens = [f"<replaycad-{concept_id}-{index}>" for index in range(self.config.semantic_tokens)]
        known_vocabulary = tokenizer.get_vocab()
        missing = [token for token in tokens if token not in known_vocabulary]
        if missing:
            added = tokenizer.add_tokens(missing)
            if added != len(missing):
                raise RuntimeError(f"Tokenizer refused {len(missing) - added} ReplayCAD placeholder token(s)")
            text_encoder.resize_token_embeddings(len(tokenizer))

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        embedding_layer = text_encoder.get_input_embeddings()
        with torch.no_grad():
            if self.config.initializer_word is None:
                generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
                init = torch.rand(
                    self.config.semantic_tokens,
                    embedding_layer.weight.shape[1],
                    generator=generator,
                ).to(device=embedding_layer.weight.device, dtype=embedding_layer.weight.dtype)
            else:
                word_ids = tokenizer(self.config.initializer_word, add_special_tokens=False)["input_ids"]
                if len(word_ids) != 1:
                    raise ValueError(
                        f"initializer_word '{self.config.initializer_word}' must map to exactly one token, "
                        f"got {len(word_ids)}"
                    )
                init = embedding_layer.weight[word_ids[0]].unsqueeze(0).repeat(self.config.semantic_tokens, 1)
            embedding_layer.weight[token_ids] = init

        self._placeholder_ids = list(token_ids)
        return list(token_ids)

    def _conditioning(self, pipeline, prompts: list[str]) -> torch.Tensor:
        """Encode prompts into the U-Net's cross-attention conditioning.

        ``max_length`` uses the text encoder's own position-embedding limit, not
        ``tokenizer.model_max_length`` -- for LDM-Bert those disagree (77 vs. an effectively
        unbounded default) and exceeding 77 raises ``IndexError``. The encoded dict is also
        filtered to the keys the encoder's ``forward`` accepts, since ``BertTokenizer`` emits
        ``token_type_ids`` that ``LDMBertModel.forward`` doesn't take. Both are no-ops for Stable
        Diffusion/CLIP.
        """
        tokenizer, text_encoder = pipeline.tokenizer, self._text_encoder(pipeline)
        placeholders = " ".join(tokenizer.convert_ids_to_tokens(self._placeholder_ids) if self._placeholder_ids else [])
        texts = [prompt.replace("*", placeholders) for prompt in prompts]

        max_length = getattr(text_encoder.config, "max_position_embeddings", tokenizer.model_max_length)
        encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

        accepted = set(inspect.signature(text_encoder.forward).parameters) & set(tokenizer.model_input_names)
        encoded = {key: value.to(text_encoder.device) for key, value in encoded.items() if key in accepted}
        return text_encoder(**encoded)[0]

    def _encode(self, pipeline, batch: torch.Tensor) -> torch.Tensor:
        """Encode NCHW images in [-1, 1] into scaled latents."""
        vae = self._vae(pipeline)
        posterior = vae.encode(batch)
        latents = posterior.latent_dist.sample() if hasattr(posterior, "latent_dist") else posterior.latents
        return latents * self.config.latent_scaling_factor

    @staticmethod
    def _pad_unconditional(uncond: torch.Tensor, target_tokens: int) -> torch.Tensor:
        """Left-pad the unconditional embedding with zero tokens to ``target_tokens``.

        Matches ``ldm/models/diffusion/ddim.py:183-184``: the zero tokens participate in the
        unconditional branch's cross-attention softmax, so dropping them changes the guidance
        signal, not just the shape. The pad goes in front, as it does there.
        """
        missing = target_tokens - uncond.shape[1]
        if missing <= 0:
            return uncond
        padding = torch.zeros(uncond.shape[0], missing, uncond.shape[2], device=uncond.device, dtype=uncond.dtype)
        return torch.cat([padding, uncond], dim=1)

    def _sample_timesteps(self, pipeline, batch_size: int, device) -> torch.Tensor:
        """Draw training timesteps as in the released implementation (ddpm2.py:984).

        ``uniform`` matches ``average_t=True`` (the class default); ``cubic`` matches
        ``average_t=False`` -- ``probs = arange(T) ** 3`` normalized -- used only for pcb1-4 and fryum.
        """
        num_train_timesteps = pipeline.scheduler.config.num_train_timesteps
        if self.config.timestep_sampling == "uniform":
            return torch.randint(0, num_train_timesteps, (batch_size,), device=device, dtype=torch.long)

        probabilities = torch.arange(0, num_train_timesteps, dtype=torch.float) ** 3
        probabilities = probabilities / probabilities.sum()
        return torch.multinomial(probabilities, batch_size, replacement=True).to(device=device, dtype=torch.long)

    @staticmethod
    def _to_model_input(array: np.ndarray, device, dtype) -> torch.Tensor:
        """(N, H, W, C) uint8 -> (N, C, H, W) float in [-1, 1], the LDM convention."""
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32) / 127.5 - 1.0)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
        return tensor.permute(0, 3, 1, 2).to(device=device, dtype=dtype)

    @staticmethod
    def _install_substitution_hook(embedding_layer, token_ids: list[int], placeholder: torch.Tensor):
        """Register a forward hook substituting ``placeholder``'s rows for ``token_ids``' rows in
        the embedding layer's output; ``embedding_layer.weight`` itself is never written.

        The caller must remove the returned handle, including on failure: hooks stack rather than
        replace, and the embedding layer is shared across concepts, so a leaked hook would keep
        substituting this concept's stale rows into every later concept's forward passes.
        """
        vocab_size = embedding_layer.weight.shape[0]
        placeholder_row = torch.full((vocab_size,), -1, dtype=torch.long, device=embedding_layer.weight.device)
        placeholder_row[torch.as_tensor(token_ids, device=embedding_layer.weight.device)] = torch.arange(
            len(token_ids), device=embedding_layer.weight.device
        )

        def _substitute_placeholders(module, args, kwargs, output):
            input_ids = args[0] if args else kwargs["input"]
            row = placeholder_row[input_ids]
            is_placeholder = row >= 0
            if not bool(is_placeholder.any()):
                return output
            substituted = placeholder[row.clamp(min=0)].to(output.dtype)
            return torch.where(is_placeholder.unsqueeze(-1), substituted, output)

        return embedding_layer.register_forward_hook(_substitute_placeholders, with_kwargs=True)

    def compress(self, concept_id: str, images: np.ndarray, masks: np.ndarray):
        pipeline = self._load()
        if len(images) == 0:
            raise ValueError(f"Cannot compress empty concept '{concept_id}'")
        # A zero-length mask array means spatial conditioning is off; any other mismatch is a bug.
        if len(masks) != 0 and len(images) != len(masks):
            raise ValueError(f"ReplayCAD got {len(images)} images and {len(masks)} masks for '{concept_id}'")

        torch.manual_seed(self.config.seed)
        token_ids = self._install_placeholders(pipeline, concept_id)
        text_encoder = self._text_encoder(pipeline)
        embedding_layer = text_encoder.get_input_embeddings()

        placeholder = torch.nn.Parameter(embedding_layer.weight[token_ids].detach().clone())
        hook = self._install_substitution_hook(embedding_layer, token_ids, placeholder)

        try:
            projection = None
            parameter_groups = [{"params": [placeholder], "lr": self.config.semantic_learning_rate}]
            if self.config.use_spatial_conditioning:
                projection = MaskProjection(
                    self.config.mask_group_width, self.config.mask_projection_width, self.config.condition_dim
                ).to(device=embedding_layer.weight.device, dtype=embedding_layer.weight.dtype)
                parameter_groups.append(
                    {"params": list(projection.parameters()), "lr": self.config.spatial_learning_rate}
                )
            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=self.config.weight_decay)

            rng = np.random.default_rng(self.config.seed)
            device, dtype = embedding_layer.weight.device, embedding_layer.weight.dtype
            batch_size = min(self.config.compression_batch_size, len(images))

            self._run_compression_loop(
                pipeline, projection, optimizer, images, masks, rng, device, dtype, batch_size, concept_id
            )
        finally:
            hook.remove()

        learned = placeholder.detach().cpu().clone()
        projection_state = (
            None if projection is None else {k: v.detach().cpu() for k, v in projection.state_dict().items()}
        )
        return learned, projection_state

    def _run_compression_loop(
        self, pipeline, projection, optimizer, images, masks, rng, device, dtype, batch_size, concept_id
    ):
        for step in range(self.config.compression_steps):
            indices = rng.choice(len(images), size=batch_size, replace=len(images) < batch_size)
            if projection is not None:
                pairs = [
                    augment_training_pair(images[index], masks[index], self.config.train_augmentation, rng)
                    for index in indices
                ]
                augmented_images = [pair[0] for pair in pairs]
            else:
                augmented_images = [
                    augment_training_pair(images[index], images[index], self.config.train_augmentation, rng)[0]
                    for index in indices
                ]
            image_batch = self._to_model_input(np.stack(augmented_images), device, dtype)
            latents = self._encode(pipeline, image_batch)

            noise = torch.randn_like(latents)
            timesteps = self._sample_timesteps(pipeline, batch_size, device)
            noisy = pipeline.scheduler.add_noise(latents, noise, timesteps)

            prompts = [str(rng.choice(self.config.prompt_templates)).format("*") for _ in range(batch_size)]
            conditioning = self._conditioning(pipeline, prompts)
            if projection is not None:
                mask_batch = self._to_model_input(np.stack([pair[1] for pair in pairs]), device, dtype)
                mask_latents = self._encode(pipeline, mask_batch)
                conditioning = torch.cat([conditioning, projection(mask_latents)], dim=1)

            predicted = pipeline.unet(noisy, timesteps, encoder_hidden_states=conditioning).sample
            loss = torch.nn.functional.mse_loss(predicted.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                logger.info(
                    "ReplayCAD compression %s step %d/%d loss %.5f",
                    concept_id,
                    step,
                    self.config.compression_steps,
                    loss.item(),
                )

    def generate(self, artifact: ConceptArtifact, count: int, seed: int) -> np.ndarray:
        pipeline = self._load()
        token_ids = self._install_placeholders(pipeline, artifact.concept_id)
        text_encoder = self._text_encoder(pipeline)
        embedding_layer = text_encoder.get_input_embeddings()
        device, dtype = embedding_layer.weight.device, embedding_layer.weight.dtype

        with torch.no_grad():
            embedding_layer.weight[token_ids] = artifact.embedding.to(device=device, dtype=dtype)

        projection = None
        if artifact.projection_state is not None:
            projection = MaskProjection(
                self.config.mask_group_width, self.config.mask_projection_width, self.config.condition_dim
            ).to(device=device, dtype=dtype)
            projection.load_state_dict(artifact.projection_state)
            projection.eval()

        pipeline.scheduler.set_timesteps(self.config.inference_steps, device=device)
        rng = np.random.default_rng(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent_side = self.config.resolution // 8

        produced = []
        with torch.no_grad():
            while sum(len(chunk) for chunk in produced) < count:
                batch = min(self.config.generation_batch_size, count - sum(len(c) for c in produced))

                conditioning = self._conditioning(pipeline, [self.config.generation_prompt] * batch)
                if projection is not None and len(artifact.masks):
                    chosen = np.stack(
                        [
                            augment_mask(artifact.masks[rng.integers(len(artifact.masks))], self.config, rng)
                            for _ in range(batch)
                        ]
                    )
                    mask_latents = self._encode(pipeline, self._to_model_input(chosen, device, dtype))
                    conditioning = torch.cat([conditioning, projection(mask_latents)], dim=1)

                uncond = self._conditioning(pipeline, [""] * batch)
                uncond = self._pad_unconditional(uncond, conditioning.shape[1])

                latents = torch.randn(
                    (batch, pipeline.unet.config.in_channels, latent_side, latent_side), generator=generator
                ).to(device=device, dtype=dtype)
                latents = latents * pipeline.scheduler.init_noise_sigma

                for timestep in pipeline.scheduler.timesteps:
                    model_input = pipeline.scheduler.scale_model_input(latents, timestep)
                    noise_cond = pipeline.unet(model_input, timestep, encoder_hidden_states=conditioning).sample
                    noise_uncond = pipeline.unet(model_input, timestep, encoder_hidden_states=uncond).sample
                    guided = noise_uncond + self.config.guidance_scale * (noise_cond - noise_uncond)
                    latents = pipeline.scheduler.step(
                        guided, timestep, latents, eta=self.config.ddim_eta, generator=generator
                    ).prev_sample

                decoded = self._vae(pipeline).decode(latents / self.config.latent_scaling_factor).sample
                decoded = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).round()
                produced.append(decoded.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy())

        return np.concatenate(produced, axis=0)[:count]
