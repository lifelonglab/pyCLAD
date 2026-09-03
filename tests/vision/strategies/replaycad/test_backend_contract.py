import numpy as np
import pytest
import torch

from pyclad.vision.strategies.replaycad.artifacts import ConceptArtifact
from pyclad.vision.strategies.replaycad.backend import DiffusersBackend


def test_stub_matches_the_declared_shapes(stub_backend, replaycad_config):
    config = replaycad_config()
    backend = stub_backend(config)
    images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    masks = np.full((2, 8, 8), 255, dtype=np.uint8)

    embedding, projection_state = backend.compress("screw", images, masks)

    assert embedding.shape == (config.semantic_tokens, config.condition_dim)
    assert set(projection_state) == {"layers.0.weight", "layers.0.bias"}


def test_stub_generate_returns_uint8_images(stub_backend, replaycad_config):
    config = replaycad_config()
    backend = stub_backend(config)
    artifact = ConceptArtifact(
        concept_id="screw",
        embedding=torch.zeros(config.semantic_tokens, config.condition_dim),
        projection_state=None,
        masks=np.full((1, 8, 8), 255, dtype=np.uint8),
        config_hash="deadbeef",
        spatial_tokens=1,
    )

    images = backend.generate(artifact, count=3, seed=0)

    assert images.shape == (3, 8, 8, 3)
    assert images.dtype == np.uint8


def test_stub_rejects_misaligned_masks(stub_backend):
    backend = stub_backend()

    with pytest.raises(ValueError, match="1 masks"):
        backend.compress("screw", np.zeros((2, 8, 8, 3), dtype=np.uint8), np.zeros((1, 8, 8), dtype=np.uint8))


def test_placeholder_installation_is_idempotent(replaycad_config):
    """A concept is installed again every time it is replayed, so this must not fail or duplicate."""
    config = replaycad_config()
    backend = DiffusersBackend(config)

    class FakeTokenizer:
        def __init__(self):
            self.vocabulary = {}

        def get_vocab(self):
            return dict(self.vocabulary)

        def add_tokens(self, tokens):
            added = 0
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
                    added += 1
            return added

        def convert_tokens_to_ids(self, tokens):
            return [self.vocabulary[token] for token in tokens]

        def __len__(self):
            return len(self.vocabulary)

    class FakeTextEncoder:
        def __init__(self, condition_dim):
            self.embedding = torch.nn.Embedding(64, condition_dim)
            self.resize_calls = 0

        def resize_token_embeddings(self, size):
            self.resize_calls += 1

        def get_input_embeddings(self):
            return self.embedding

    class FakePipeline:
        def __init__(self, condition_dim):
            self.tokenizer = FakeTokenizer()
            self.text_encoder = FakeTextEncoder(condition_dim)

    pipeline = FakePipeline(config.condition_dim)

    first = backend._install_placeholders(pipeline, "screw")
    second = backend._install_placeholders(pipeline, "screw")

    assert first == second
    assert len(pipeline.tokenizer.vocabulary) == config.semantic_tokens
    assert pipeline.text_encoder.resize_calls == 1  # the table grows once, not on every replay


def test_install_placeholders_does_not_disturb_the_global_rng(replaycad_config):
    """torch.manual_seed() here would pin the *global* RNG to config.seed on every call --
    including every replay in generate() -- so anything downstream that draws from the global RNG
    without its own generator (e.g. _encode's VAE posterior sample for the mask latent) would get
    the same "random" draw regardless of the per-concept seed generate() was given. Placeholder
    init must get its randomness from a local generator instead, leaving global state untouched.
    """
    config = replaycad_config()
    backend = DiffusersBackend(config)

    class FakeTokenizer:
        def __init__(self):
            self.vocabulary = {}

        def get_vocab(self):
            return dict(self.vocabulary)

        def add_tokens(self, tokens):
            added = 0
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
                    added += 1
            return added

        def convert_tokens_to_ids(self, tokens):
            return [self.vocabulary[token] for token in tokens]

        def __len__(self):
            return len(self.vocabulary)

    class FakeTextEncoder:
        def __init__(self, condition_dim):
            self.embedding = torch.nn.Embedding(64, condition_dim)

        def resize_token_embeddings(self, size):
            return None

        def get_input_embeddings(self):
            return self.embedding

    class FakePipeline:
        def __init__(self, condition_dim):
            self.tokenizer = FakeTokenizer()
            self.text_encoder = FakeTextEncoder(condition_dim)

    pipeline = FakePipeline(config.condition_dim)

    torch.manual_seed(123)  # arbitrary unrelated prior global state
    before = torch.get_rng_state()

    backend._install_placeholders(pipeline, "screw")

    assert torch.equal(before, torch.get_rng_state())


def test_unconditional_padding_is_prepended(replaycad_config):
    config = replaycad_config()
    uncond = torch.ones(2, 5, config.condition_dim)

    padded = DiffusersBackend._pad_unconditional(uncond, target_tokens=8)

    assert padded.shape == (2, 8, config.condition_dim)
    # The released sampler puts the zeros first (ddim.py:184), leaving the real text tokens last.
    assert torch.all(padded[:, :3] == 0)
    assert torch.all(padded[:, 3:] == 1)


def test_unconditional_padding_is_a_no_op_when_lengths_match(replaycad_config):
    uncond = torch.ones(2, 8, 4)

    assert DiffusersBackend._pad_unconditional(uncond, target_tokens=8) is uncond


def test_substitution_hook_is_removed_when_compression_raises(replaycad_config, monkeypatch):
    """A leaked forward hook would keep substituting this concept's stale placeholder rows into
    the next concept's forward passes -- the embedding layer is the same module for every concept
    on a cached pipeline -- silently corrupting that concept's conditioning.
    """
    config = replaycad_config(use_spatial_conditioning=False)
    backend = DiffusersBackend(config)

    class FakeTokenizer:
        def __init__(self):
            self.vocabulary = {}

        def get_vocab(self):
            return dict(self.vocabulary)

        def add_tokens(self, tokens):
            for token in tokens:
                self.vocabulary.setdefault(token, len(self.vocabulary))
            return len(tokens)

        def convert_tokens_to_ids(self, tokens):
            return [self.vocabulary[token] for token in tokens]

        def __len__(self):
            return len(self.vocabulary)

    class FakeTextEncoder:
        def __init__(self, condition_dim):
            self.embedding = torch.nn.Embedding(64, condition_dim)

        def resize_token_embeddings(self, size):
            return None

        def get_input_embeddings(self):
            return self.embedding

    class ExplodingVae:
        def encode(self, batch):
            raise RuntimeError("simulated out of memory")

    class FakePipeline:
        def __init__(self, condition_dim):
            self.tokenizer = FakeTokenizer()
            self.text_encoder = FakeTextEncoder(condition_dim)
            self.vae = ExplodingVae()

    pipeline = FakePipeline(config.condition_dim)
    monkeypatch.setattr(backend, "_load", lambda: pipeline)
    embedding_layer = pipeline.text_encoder.embedding

    with pytest.raises(RuntimeError, match="simulated out of memory"):
        backend.compress(
            "screw",
            np.zeros((1, config.resolution, config.resolution, 3), dtype=np.uint8),
            np.zeros((1, config.resolution, config.resolution), dtype=np.uint8),
        )

    assert not embedding_layer._forward_hooks


def test_substitution_hook_is_removed_when_projection_construction_raises(replaycad_config, monkeypatch):
    """The `try` guarding `hook.remove()` used to open only around `_run_compression_loop`,
    leaving the hook registration, `MaskProjection(...).to(device=...)` and the AdamW construction
    unguarded in between. A CUDA OOM in that gap is realistic -- straight after a full pipeline was
    just loaded onto the same device -- and would leak the hook exactly like a loop failure does:
    the leaked closure keeps capturing this call's *aborted* `placeholder`, so every later
    generate() for this concept would silently substitute mid-training values instead of the
    artifact's actually-stored embedding. Unlike
    test_substitution_hook_is_removed_when_compression_raises above, this one fails inside the gap
    itself (spatial conditioning on, so MaskProjection is actually constructed), not inside the
    loop, so it would not have caught a `try` that opened too late.
    """
    config = replaycad_config(use_spatial_conditioning=True)
    backend = DiffusersBackend(config)

    class FakeTokenizer:
        def __init__(self):
            self.vocabulary = {}

        def get_vocab(self):
            return dict(self.vocabulary)

        def add_tokens(self, tokens):
            for token in tokens:
                self.vocabulary.setdefault(token, len(self.vocabulary))
            return len(tokens)

        def convert_tokens_to_ids(self, tokens):
            return [self.vocabulary[token] for token in tokens]

        def __len__(self):
            return len(self.vocabulary)

    class FakeTextEncoder:
        def __init__(self, condition_dim):
            self.embedding = torch.nn.Embedding(64, condition_dim)

        def resize_token_embeddings(self, size):
            return None

        def get_input_embeddings(self):
            return self.embedding

    class FakePipeline:
        def __init__(self, condition_dim):
            self.tokenizer = FakeTokenizer()
            self.text_encoder = FakeTextEncoder(condition_dim)

    pipeline = FakePipeline(config.condition_dim)
    monkeypatch.setattr(backend, "_load", lambda: pipeline)
    embedding_layer = pipeline.text_encoder.embedding

    import pyclad.vision.strategies.replaycad.backend as backend_module

    def exploding_to(self, *args, **kwargs):
        raise RuntimeError("simulated CUDA out of memory")

    monkeypatch.setattr(backend_module.MaskProjection, "to", exploding_to)

    with pytest.raises(RuntimeError, match="simulated CUDA out of memory"):
        backend.compress(
            "screw",
            np.zeros((1, config.resolution, config.resolution, 3), dtype=np.uint8),
            np.zeros((1, config.resolution, config.resolution), dtype=np.uint8),
        )

    assert not embedding_layer._forward_hooks


class _Config:
    """Dumb attribute bag, standing in for the small pieces of a diffusers config/module this
    file's fakes need to expose (e.g. ``scheduler.config.num_train_timesteps``)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_loop_pipeline():
    """Minimal pipeline for _run_compression_loop with _conditioning/_encode stubbed out: only
    scheduler.config.num_train_timesteps, scheduler.add_noise and unet(...) are still touched."""
    scheduler = _Config(config=_Config(num_train_timesteps=1000))
    scheduler.add_noise = lambda latents, noise, timesteps: latents + noise
    unet = lambda noisy, timesteps, encoder_hidden_states: _Config(  # noqa: E731
        sample=torch.zeros_like(noisy, requires_grad=True)
    )
    return _Config(scheduler=scheduler, unet=unet)


def test_compression_draws_an_independent_prompt_per_image(replaycad_config, monkeypatch):
    """The released dataset draws one caption per __getitem__ call, independently for every image
    (personalized.py:218). Reusing a single per-step draw across the whole batch would silently
    cut caption diversity by a factor of compression_batch_size (16x for MVTec's default). This
    drives _run_compression_loop directly with everything except the prompt draw stubbed out, and
    checks that a 6-image batch does not all get the same prompt.
    """
    config = replaycad_config(compression_steps=1, compression_batch_size=6)
    backend = DiffusersBackend(config)

    captured_prompts: list[list[str]] = []

    def fake_conditioning(pipeline, *args):
        # Tolerates both the old call shape (prompt: str, batch_size: int) and the new one
        # (prompts: list[str]), so this same test can run unmodified against either
        # _run_compression_loop and let the diversity assertion below be the real signal.
        if len(args) == 2:
            prompt, batch_size = args
            prompts = [prompt] * batch_size
        else:
            (prompts,) = args
        captured_prompts.append(list(prompts))
        return torch.zeros(len(prompts), 1, config.condition_dim)

    monkeypatch.setattr(backend, "_conditioning", fake_conditioning)
    monkeypatch.setattr(backend, "_encode", lambda pipeline, batch: torch.zeros(len(batch), 4, 2, 2))

    images = np.zeros((6, config.resolution, config.resolution, 3), dtype=np.uint8)
    # Non-empty and matching images in length: this test is about prompt diversity alone, kept
    # independent of the (separately tested) empty-masks tolerance.
    masks = np.zeros((6, config.resolution, config.resolution), dtype=np.uint8)
    rng = np.random.default_rng(0)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)

    backend._run_compression_loop(
        _fake_loop_pipeline(), None, optimizer, images, masks, rng, torch.device("cpu"), torch.float32, 6, "screw"
    )

    assert len(captured_prompts) == 1  # one step
    assert len(captured_prompts[0]) == 6  # one prompt per image
    assert len(set(captured_prompts[0])) > 1  # not 6 copies of a single per-step draw


def test_compression_loop_tolerates_empty_masks_when_spatial_conditioning_is_off(replaycad_config, monkeypatch):
    """memory.py passes a zero-length mask array when spatial conditioning is off, to skip a full
    SAM pass whose output the backend would never use. The loop must not index into it:
    augment_training_pair only ever receives a real mask when projection is not None.
    """
    config = replaycad_config(compression_batch_size=3)
    backend = DiffusersBackend(config)

    monkeypatch.setattr(
        backend, "_conditioning", lambda pipeline, prompts: torch.zeros(len(prompts), 1, config.condition_dim)
    )
    monkeypatch.setattr(backend, "_encode", lambda pipeline, batch: torch.zeros(len(batch), 4, 2, 2))

    images = np.zeros((3, config.resolution, config.resolution, 3), dtype=np.uint8)
    masks = np.zeros((0, config.resolution, config.resolution), dtype=np.uint8)  # spatial conditioning off
    rng = np.random.default_rng(0)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)

    # Must not raise IndexError from masks[index] on the empty array.
    backend._run_compression_loop(
        _fake_loop_pipeline(), None, optimizer, images, masks, rng, torch.device("cpu"), torch.float32, 3, "screw"
    )


def test_compress_accepts_a_zero_length_mask_array(replaycad_config, monkeypatch):
    """The top-level image/mask count check must tolerate a zero-length mask array -- memory.py's
    way of saying spatial conditioning is off and no masks were computed -- while still catching a
    genuine misalignment (covered separately by test_stub_rejects_misaligned_masks's real-count
    case, and by DiffusersBackend sharing the same check).
    """
    config = replaycad_config(use_spatial_conditioning=False)
    backend = DiffusersBackend(config)

    class FakeTokenizer:
        def get_vocab(self):
            return {}

        def add_tokens(self, tokens):
            return len(tokens)

        def convert_tokens_to_ids(self, tokens):
            return list(range(len(tokens)))

        def __len__(self):
            return config.semantic_tokens

    class FakeTextEncoder:
        def __init__(self):
            self.embedding = torch.nn.Embedding(64, config.condition_dim)

        def resize_token_embeddings(self, size):
            return None

        def get_input_embeddings(self):
            return self.embedding

    class FakePipeline:
        tokenizer = FakeTokenizer()
        text_encoder = FakeTextEncoder()

    monkeypatch.setattr(backend, "_load", lambda: FakePipeline())
    monkeypatch.setattr(backend, "_run_compression_loop", lambda *args, **kwargs: None)

    images = np.zeros((2, config.resolution, config.resolution, 3), dtype=np.uint8)
    masks = np.zeros((0, config.resolution, config.resolution), dtype=np.uint8)

    embedding, projection_state = backend.compress("screw", images, masks)

    assert embedding.shape == (config.semantic_tokens, config.condition_dim)
    assert projection_state is None


# --- placeholder substitution: vocabulary integrity, gradient flow, weight_decay, id->row mapping
#
# DiffusersBackend.compress() used to hand embedding_layer.weight straight to AdamW and zero the
# gradient of every non-placeholder row with a backward hook, which only worked because
# weight_decay was forced to 0 -- AdamW's decoupled decay ignores the gradient mask and would
# otherwise corrode the whole vocabulary. It now holds the K learned vectors in their own
# nn.Parameter (`placeholder`) and substitutes them into the embedding layer's *output* via a
# forward hook, leaving embedding_layer.weight untouched and requires_grad=False throughout, so a
# real weight_decay (matching the released implementation's implicit PyTorch AdamW default,
# ddpm2.py:1563-1573) can no longer touch it. The first three tests below drive that mechanism
# through a real compress() call: a minimal "text encoder" that is just its own token-embedding
# lookup, so the loss is a genuine (if physically meaningless) differentiable function of the
# substituted output -- and therefore of `placeholder` -- with no transformer internals to fake.
#
# None of those three checks *which* row lands where, though. Confirmed by mutation: rotating the
# id->row mapping by one, and making every placeholder id gather row 0 instead of its own, both
# left every test in this file green except one that removes substitution outright -- because
# _DifferentiableUnet's original reduction (plain mean over the token axis, then sum) is invariant
# under permuting *which* row lands at which position, so those mutants produced the exact same
# loss trajectory as the correct mapping. _DifferentiableUnet below now weights the token axis by
# position before reducing, so *where* a row lands actually matters to the loss; and
# test_substitution_hook_maps_each_placeholder_id_to_its_own_row further down asserts the mapping
# directly, independent of any loss or gradient.


class _DifferentiableTokenizer:
    """Vocabulary starts with one reserved pad entry (id 0), so the padding `_conditioning`
    relies on can never collide with a newly added placeholder id."""

    def __init__(self):
        self.vocabulary = {"<pad>": 0}
        self.model_input_names = ["input_ids"]
        self.model_max_length = 8

    def get_vocab(self):
        return dict(self.vocabulary)

    def add_tokens(self, tokens):
        added = 0
        for token in tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
                added += 1
        return added

    def convert_tokens_to_ids(self, tokens):
        return [self.vocabulary[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        reverse = {index: token for token, index in self.vocabulary.items()}
        return [reverse[index] for index in ids]

    def __len__(self):
        return len(self.vocabulary)

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert padding == "max_length" and truncation and return_tensors == "pt"
        pad_id = self.vocabulary["<pad>"]
        rows = []
        for text in texts:
            ids = [self.vocabulary[piece] for piece in text.split(" ") if piece]
            ids = (ids + [pad_id] * max_length)[:max_length]
            rows.append(ids)
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}


class _DifferentiableTextEncoder:
    """A "text encoder" that is just its own token embedding lookup: text_encoder(input_ids) ==
    embedding_layer(input_ids), the exact call the substitution hook attaches to. Standing in for
    the real CLIP/LDMBert encoders, which do the same lookup internally before further layers that
    are irrelevant to whether the substitution and its gradient are wired correctly.
    """

    def __init__(self, condition_dim):
        self.embedding = torch.nn.Embedding(64, condition_dim)
        self.config = _Config()
        self.device = torch.device("cpu")

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, size):
        return None

    def forward(self, input_ids=None):
        return (self.embedding(input_ids),)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class _DifferentiableUnet:
    """Returns noisy + a bias derived from encoder_hidden_states, so the MSE loss is a real
    differentiable function of the conditioning tensor -- and, through the substitution hook, of
    `placeholder` -- without needing an actual denoiser.

    The token axis is weighted by position (1, 2, ..., L) before reducing, rather than a plain
    mean: a plain mean-then-sum is invariant under permuting which row lands at which position, so
    it cannot distinguish a correct id->row mapping from a rotated one -- both produce the exact
    same loss trajectory (see the module comment above this class). Position weighting makes
    *where* a row lands actually change the loss.
    """

    def __call__(self, noisy, timesteps, encoder_hidden_states):
        length = encoder_hidden_states.shape[1]
        position_weight = torch.arange(
            1, length + 1, dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device
        ).view(1, -1, 1)
        bias = (encoder_hidden_states * position_weight).mean(dim=1).sum(dim=-1).view(-1, 1, 1, 1)
        return _Config(sample=noisy + bias)


def _build_differentiable_pipeline(config):
    scheduler = _Config(config=_Config(num_train_timesteps=1000))
    scheduler.add_noise = lambda latents, noise, timesteps: latents + noise
    return _Config(
        tokenizer=_DifferentiableTokenizer(),
        text_encoder=_DifferentiableTextEncoder(config.condition_dim),
        unet=_DifferentiableUnet(),
        scheduler=scheduler,
    )


def _wire_differentiable_backend(config, monkeypatch):
    """Builds the pipeline above, points a fresh backend at it, and stubs out only image encoding
    (the VAE round trip is covered separately by test_real_diffusion.py and is irrelevant to the
    substitution mechanism under test here)."""
    backend = DiffusersBackend(config)
    pipeline = _build_differentiable_pipeline(config)
    monkeypatch.setattr(backend, "_load", lambda: pipeline)
    monkeypatch.setattr(backend, "_encode", lambda pipeline, batch: torch.zeros(len(batch), 4, 2, 2))
    return backend, pipeline


def test_compression_leaves_the_vocabulary_bit_identical(replaycad_config, monkeypatch):
    """The property the old gradient-masking backward hook existed to protect. Confirmed
    load-bearing by temporarily making the substitution write into embedding_layer.weight too and
    watching this test fail -- see optimizer-fix-report.md for the transcript.
    """
    config = replaycad_config(use_spatial_conditioning=False, prompt_templates=("*",), compression_steps=2)
    backend, pipeline = _wire_differentiable_backend(config, monkeypatch)

    token_ids = backend._install_placeholders(pipeline, "screw")
    embedding_layer = pipeline.text_encoder.get_input_embeddings()
    before = embedding_layer.weight.detach().clone()

    images = np.zeros((config.compression_batch_size, config.resolution, config.resolution, 3), dtype=np.uint8)
    masks = np.zeros((0, config.resolution, config.resolution), dtype=np.uint8)
    backend.compress("screw", images, masks)

    after = embedding_layer.weight.detach()
    non_placeholder = torch.ones(after.shape[0], dtype=torch.bool)
    non_placeholder[token_ids] = False

    assert non_placeholder.sum() > 0  # the fake vocabulary has rows besides the placeholders
    assert torch.equal(before[non_placeholder], after[non_placeholder])


def test_compression_moves_the_placeholder_parameter(replaycad_config, monkeypatch):
    """Gradients must still reach the learned embedding -- the whole reason to keep optimizing it
    -- even though it is no longer a slice of embedding_layer.weight.
    """
    config = replaycad_config(use_spatial_conditioning=False, prompt_templates=("*",), compression_steps=2)
    backend, pipeline = _wire_differentiable_backend(config, monkeypatch)

    token_ids = backend._install_placeholders(pipeline, "screw")
    embedding_layer = pipeline.text_encoder.get_input_embeddings()
    initial = embedding_layer.weight[token_ids].detach().clone()

    images = np.zeros((config.compression_batch_size, config.resolution, config.resolution, 3), dtype=np.uint8)
    masks = np.zeros((0, config.resolution, config.resolution), dtype=np.uint8)
    learned, _ = backend.compress("screw", images, masks)

    assert learned.shape == initial.shape
    assert not torch.allclose(learned, initial)


def test_substitution_hook_maps_each_placeholder_id_to_its_own_row(replaycad_config, monkeypatch):
    """Direct assertion on the id->row mapping itself, independent of any loss or gradient --
    install the real hook (DiffusersBackend._install_substitution_hook, the same code compress()
    uses) with `placeholder` set to K distinguishable rows, call the embedding layer on exactly the
    placeholder ids, and check each output row against its own placeholder row one at a time.

    Confirmed load-bearing by mutation: rotating the id->row mapping by one, and making every
    placeholder id gather row 0 instead of its own, both slip past every *other* test in this file
    (see the module comment above _DifferentiableUnet) -- this test catches both, because it never
    goes through a loss at all.
    """
    config = replaycad_config()
    backend, pipeline = _wire_differentiable_backend(config, monkeypatch)

    token_ids = backend._install_placeholders(pipeline, "screw")
    text_encoder = pipeline.text_encoder
    embedding_layer = text_encoder.get_input_embeddings()

    num_tokens, condition_dim = config.semantic_tokens, config.condition_dim
    # arange, not random init: guarantees every row is bitwise distinguishable from every other,
    # so a mismapped row can never accidentally equal the row it should have been compared against.
    placeholder = torch.arange(num_tokens * condition_dim, dtype=torch.float32).reshape(num_tokens, condition_dim)

    hook = backend._install_substitution_hook(embedding_layer, token_ids, placeholder)
    try:
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        output = text_encoder(input_ids=input_ids)[0]
    finally:
        hook.remove()

    assert output.shape == (1, num_tokens, condition_dim)
    for index in range(num_tokens):
        assert torch.equal(output[0, index], placeholder[index])


def test_compression_optimizer_uses_the_configured_weight_decay(replaycad_config, monkeypatch):
    """No group-specific override: compress() passes one scalar weight_decay to AdamW's
    constructor, exactly as ddpm2.py:1563-1573 does (there, by omission -- PyTorch's own default).
    """
    config = replaycad_config(use_spatial_conditioning=False, prompt_templates=("*",), compression_steps=1)
    assert config.weight_decay == pytest.approx(1e-2)
    backend, pipeline = _wire_differentiable_backend(config, monkeypatch)

    real_adamw = torch.optim.AdamW
    constructed = []

    def _recording_adamw(*args, **kwargs):
        optimizer = real_adamw(*args, **kwargs)
        constructed.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", _recording_adamw)

    images = np.zeros((config.compression_batch_size, config.resolution, config.resolution, 3), dtype=np.uint8)
    masks = np.zeros((0, config.resolution, config.resolution), dtype=np.uint8)
    backend.compress("screw", images, masks)

    assert len(constructed) == 1
    weight_decays = [group["weight_decay"] for group in constructed[0].param_groups]
    assert weight_decays == [pytest.approx(1e-2)]


def test_real_backend_reports_a_helpful_error_without_the_extra(replaycad_config, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("diffusers"):
            raise ImportError("no diffusers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(ImportError, match="replaycad"):
        DiffusersBackend(replaycad_config())._load()


# --- C1: the LDM-256 conditioning path (BertTokenizer + LDMBertModel) ------------------------
#
# Stable Diffusion's CLIPTokenizer/CLIPTextModel survive _conditioning()'s naive
# tokenizer(**kwargs) -> text_encoder(**encoded) plumbing by coincidence: CLIPTokenizer never
# produces token_type_ids, and its checkpoints ship a tokenizer_config.json capping
# model_max_length at exactly 77. Neither coincidence holds for the LDM-256 profile that MVTec
# uses by default. Verified against the installed diffusers 0.40 / transformers 5.16 rather than
# assumed:
#   - transformers.models.bert.tokenization_bert.BertTokenizer.model_input_names includes
#     "token_type_ids".
#   - diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion.LDMBertModel.forward's
#     signature is (input_ids, attention_mask, position_ids, head_mask, inputs_embeds,
#     output_attentions, output_hidden_states, return_dict) -- no token_type_ids, and no
#     **kwargs to swallow one. Calling it with token_type_ids=... raises
#     "TypeError: forward() got an unexpected keyword argument 'token_type_ids'".
#   - LDMBertConfig.max_position_embeddings defaults to 77, and LDMBertEncoder.forward looks up
#     one position row per input token from an nn.Embedding(max_position_embeddings, ...) table
#     with no bounds check of its own -- feeding it BertTokenizer's own model_max_length (512, or
#     transformers' VERY_LARGE_INTEGER sentinel when the checkpoint ships no
#     tokenizer_config.json) raises "IndexError: index out of range in self".
# The two stubs below mirror both real signatures exactly rather than special-casing the bad
# inputs, so they reject/error for the same structural reasons the real classes do.


class _LDMBertStyleConfig:
    def __init__(self, max_position_embeddings: int):
        self.max_position_embeddings = max_position_embeddings


class _LDMBertStyleEncoder:
    """Mirrors LDMBertModel: no token_type_ids parameter, and a position table that only goes up
    to max_position_embeddings."""

    def __init__(self, max_position_embeddings: int = 77):
        self.config = _LDMBertStyleConfig(max_position_embeddings)
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise IndexError("index out of range in self")
        return (torch.zeros(input_ids.shape[0], input_ids.shape[1], 8),)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class _BertStyleTokenizer:
    """Mirrors BertTokenizer's output contract: model_input_names and the padded shape it
    produces. ``model_input_names`` defaults to Bert's real list (including token_type_ids);
    pass a shorter one to mimic a tokenizer that does not produce it."""

    def __init__(self, model_input_names=("input_ids", "token_type_ids", "attention_mask"), model_max_length=512):
        self.model_input_names = list(model_input_names)
        self.model_max_length = model_max_length

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert padding == "max_length" and truncation and return_tensors == "pt"
        batch = len(texts)
        values = {
            "input_ids": torch.ones(batch, max_length, dtype=torch.long),
            "token_type_ids": torch.zeros(batch, max_length, dtype=torch.long),
            "attention_mask": torch.ones(batch, max_length, dtype=torch.long),
        }
        return {key: values[key] for key in self.model_input_names}


class _StubLDMPipeline:
    def __init__(self, tokenizer, text_encoder):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder


def test_conditioning_reproduces_bert_and_ldmbert_defaults(replaycad_config):
    """The exact combination that crashes the default MVTec/LDM-256 profile deterministically:
    BertTokenizer's real defaults (token_type_ids in model_input_names, model_max_length=512)
    against LDMBertModel's real default (max_position_embeddings=77, no token_type_ids
    parameter). Before the fix this raised TypeError on the first compression/generation step.
    """
    backend = DiffusersBackend(replaycad_config())
    pipeline = _StubLDMPipeline(tokenizer=_BertStyleTokenizer(), text_encoder=_LDMBertStyleEncoder())

    conditioning = backend._conditioning(pipeline, ["a photo of a *", "a rendering of a *"])

    assert conditioning.shape == (2, 77, 8)


def test_conditioning_bounds_length_to_the_text_encoders_own_limit(replaycad_config):
    """Isolates the length bound from the key-filtering fix: this tokenizer produces no
    token_type_ids at all, so only max_length is under test. Before the fix, max_length came
    from tokenizer.model_max_length (512) instead of the encoder's own 77-row position table,
    raising IndexError.
    """
    backend = DiffusersBackend(replaycad_config())
    pipeline = _StubLDMPipeline(
        tokenizer=_BertStyleTokenizer(model_input_names=["input_ids", "attention_mask"], model_max_length=512),
        text_encoder=_LDMBertStyleEncoder(max_position_embeddings=77),
    )

    conditioning = backend._conditioning(pipeline, ["a photo of a *"])

    assert conditioning.shape == (1, 77, 8)
