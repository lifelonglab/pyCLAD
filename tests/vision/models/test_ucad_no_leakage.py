"""Executable guarantees that UCAD never uses test data or test labels.

The reference implementation (https://github.com/shirowalker/UCAD) picks the prompt and the
knowledge bank by the best test AUROC over 25 epochs, reports a running ensemble of scores
across epochs, min-max normalises over the whole test set, and routes one task id per test set.
Each test below pins the property that makes one of those impossible here.
"""

import inspect
from typing import Iterator, Tuple

import numpy as np
import pytest
import torch

from pyclad.vision.models.ucad.config import UCADConfig
from pyclad.vision.models.ucad.ucad import UCAD
from pyclad.vision.strategies.ucad.strategy import UCADStrategy

pytest.importorskip("timm")


def _config(**overrides) -> UCADConfig:
    defaults = dict(
        backbone_name="vit_tiny_patch16_224",
        pretrained_backbone=False,
        input_size=(32, 32),
        input_range="float01",
        batch_size=2,
        feature_layer=3,
        prompt_depth=3,
        target_embed_dimension=16,
        key_size=3,
        knowledge_size=3,
        coreset_projection_dimension=4,
        coreset_starting_points=2,
        epochs=1,
        seed=0,
    )
    defaults.update(overrides)
    return UCADConfig(**defaults)


def _images(count: int, fill: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((count, 32, 32, 3), dtype=np.float32) * 0.1 + fill).astype(np.float32)


def _two_task_model(**overrides) -> UCAD:
    model = UCAD(_config(**overrides))
    model.set_current_concept("dark")
    model.fit(_images(4, 0.05, seed=1))
    model.set_current_concept("bright")
    model.fit(_images(4, 0.85, seed=2))
    return model


def _candidate_state_values(state: dict) -> Iterator[Tuple[str, object]]:
    """Yield every ``(qualified_name, value)`` reachable from a ``vars(...)`` snapshot.

    Descends into list/tuple elements, dict values, and any plain object's own ``__dict__`` --
    which is what makes ``UCADTaskMemory.key/knowledge/prompt`` (reached through
    ``model._task_memories``) and a fitted ``NearestNeighbors._fit_X`` (reached through
    ``model._key_indices``/``model._knowledge_indices``) discoverable, not just the top-level
    attributes of whatever ``state`` was passed in.

    It does not descend into ``torch.nn.Module`` instances (notably ``model.module``, the
    frozen ViT backbone): that subtree is hundreds of parameters and buffers deep, none of
    which UCAD ever refits, so walking it would add traversal cost without adding coverage.
    """

    def walk(name: str, value: object, seen: set) -> Iterator[Tuple[str, object]]:
        yield name, value
        if isinstance(value, (np.ndarray, torch.Tensor)) or isinstance(value, torch.nn.Module):
            return
        if id(value) in seen:
            return
        if isinstance(value, (list, tuple)):
            seen.add(id(value))
            for index, item in enumerate(value):
                yield from walk(f"{name}[{index}]", item, seen)
        elif isinstance(value, dict):
            seen.add(id(value))
            for key, item in value.items():
                yield from walk(f"{name}[{key!r}]", item, seen)
        elif hasattr(value, "__dict__"):
            seen.add(id(value))
            for attr_name, attr_value in vars(value).items():
                yield from walk(f"{name}.{attr_name}", attr_value, seen)

    seen: set = set()
    for name, value in state.items():
        yield from walk(name, value, seen)


def test_fit_accepts_training_data_and_nothing_else():
    # A tripwire against someone adding `test_data=` or `labels=` to reach the reference's
    # best-epoch-on-test selection.
    fit_params = inspect.signature(UCAD.fit).parameters
    assert list(fit_params) == ["self", "data"]
    # Name-only comparison above would also be satisfied by `def fit(self, **data)`, which keeps
    # the same parameter name while accepting arbitrary extra kwargs; confirm `data` is an
    # ordinary positional parameter, not a disguised catch-all.
    assert fit_params["data"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

    learn_params = inspect.signature(UCADStrategy.learn).parameters
    assert list(learn_params) == ["self", "data", "concept_id", "kwargs"]
    # Name-only comparison above would also be satisfied by an ordinary parameter literally
    # named `kwargs`; confirm it is actually the catch-all `**kwargs` and not a disguised
    # extra field that would let `learn(train, concept_id="a", kwargs={"test_data": X})` in.
    assert learn_params["kwargs"].kind == inspect.Parameter.VAR_KEYWORD


def test_learn_discards_unknown_keyword_arguments():
    # The signature tripwire above cannot fire on `**kwargs`, which is exactly the route by
    # which `learn(train, concept_id="a", test_data=X)` would arrive. Two separate properties
    # must hold for the kwargs to be harmless: they must not have changed what was learned
    # (checked below by comparing thresholds/memories against a clean run), AND they must not
    # have been kept anywhere at all. A `learn` that stashed `self._leak = kwargs` or
    # `self._model._leak = kwargs["test_data"]` would pass the comparison untouched -- nothing
    # it changed -- while still having let test data reach `fit()`. The walk below asserts
    # non-storage directly instead of inferring it from non-influence.
    data = _images(4, 0.05, seed=101)
    # 20 is distinct from every other array length used in this file (batch sizes, key/knowledge
    # sizes, per-task image counts), so a leading dimension of 20 anywhere in the learned state
    # can only have come from this leak.
    leak = _images(20, 0.95, seed=102)

    clean_strategy = UCADStrategy(UCAD(_config()))
    clean_strategy.learn(data, concept_id="dark")

    leaked_strategy = UCADStrategy(UCAD(_config()))
    leaked_strategy.learn(data, concept_id="dark", test_data=leak, labels=np.ones(len(leak)))

    clean_model, leaked_model = clean_strategy._model, leaked_strategy._model
    assert clean_model._threshold == leaked_model._threshold
    assert len(clean_model.task_memories) == len(leaked_model.task_memories) == 1
    for clean_memory, leaked_memory in zip(clean_model.task_memories, leaked_model.task_memories):
        np.testing.assert_array_equal(clean_memory.key, leaked_memory.key)
        np.testing.assert_array_equal(clean_memory.knowledge, leaked_memory.knowledge)
        torch.testing.assert_close(clean_memory.prompt, leaked_memory.prompt)

    for owner_name, owner in (("strategy", leaked_strategy), ("model", leaked_model)):
        for name, value in _candidate_state_values(vars(owner)):
            if isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim > 0 and value.shape[0] == len(leak):
                raise AssertionError(f"UCAD kept the leaked kwarg on {owner_name}.{name}")


def test_scores_do_not_depend_on_how_the_batch_is_composed():
    # The strongest structural guarantee available: if any statistic were pooled over the
    # evaluated set (the reference's per-test-set min-max normalisation and per-test-set task
    # routing), splitting or reordering the same images would move their scores.
    model = _two_task_model(batch_size=8)
    data = np.concatenate([_images(3, 0.05, seed=31), _images(3, 0.85, seed=32)])

    whole = model.predict(data).anomaly_scores
    # A constant scorer would make every assert_allclose below pass vacuously; rule it out.
    assert len(np.unique(whole)) > 1, "fixture is degenerate: scores must vary"

    model.config.batch_size = 1
    one_at_a_time = model.predict(data).anomaly_scores

    order = np.array([4, 0, 5, 2, 1, 3])
    model.config.batch_size = 8
    shuffled = model.predict(data[order]).anomaly_scores

    np.testing.assert_allclose(whole, one_at_a_time, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(whole[order], shuffled, rtol=1e-5, atol=1e-6)


def test_routing_does_not_depend_on_how_the_batch_is_composed():
    model = _two_task_model(batch_size=8)
    data = np.concatenate([_images(3, 0.05, seed=33), _images(3, 0.85, seed=34)])

    whole = model.selected_task_indices(data)

    model.config.batch_size = 1
    one_at_a_time = model.selected_task_indices(data)
    # This test's sensitivity depends on the fixture actually exercising more than one task: if
    # every image routed to the same task, a batch-pooled argmin would coincidentally agree with
    # a per-image argmin and this test would pass even against the leak it is meant to catch.
    # Assert that on `one_at_a_time`, not on `whole`: `one_at_a_time` runs one image per batch and
    # so is structurally unpoolable, whereas `whole` is exactly the batch-pooled path under test.
    # That way a real pooling leak fails on the comparison below, with an honest message, instead
    # of failing here with a misleading "fixture is degenerate".
    assert len(np.unique(one_at_a_time)) > 1, "fixture is degenerate: routing must visit more than one task"

    np.testing.assert_array_equal(whole, one_at_a_time)


def test_predict_does_not_mutate_any_learned_state():
    model = _two_task_model()
    before_threshold = model._threshold
    before_keys = [memory.key.copy() for memory in model.task_memories]
    before_knowledge = [memory.knowledge.copy() for memory in model.task_memories]
    before_prompts = [memory.prompt.clone() for memory in model.task_memories]
    # NearestNeighbors.fit(...)'s own training data -- catches a refit-on-test-features leak
    # that would leave key/knowledge/prompt themselves untouched. `_fit_X` is a private sklearn
    # attribute (no leading underscore in the public API), so a scikit-learn version bump that
    # renames or removes it turns this guarantee into a loud `AttributeError` here rather than
    # a silent pass.
    before_key_index_data = [index._fit_X.copy() for index in model._key_indices]
    before_knowledge_index_data = [index._fit_X.copy() for index in model._knowledge_indices]

    model.predict(_images(5, 0.9, seed=41))
    model.predict(_images(5, 0.02, seed=42))

    assert model._threshold == before_threshold
    assert len(model.task_memories) == 2
    for memory, key, knowledge, prompt in zip(model.task_memories, before_keys, before_knowledge, before_prompts):
        np.testing.assert_array_equal(memory.key, key)
        np.testing.assert_array_equal(memory.knowledge, knowledge)
        torch.testing.assert_close(memory.prompt, prompt)
    for index, fit_data in zip(model._key_indices, before_key_index_data):
        np.testing.assert_array_equal(index._fit_X, fit_data)
    for index, fit_data in zip(model._knowledge_indices, before_knowledge_index_data):
        np.testing.assert_array_equal(index._fit_X, fit_data)


def test_threshold_is_a_quantile_of_training_scores_only():
    model = UCAD(_config(threshold_quantile=0.9))
    train = _images(6, 0.2, seed=51)
    model.fit(train)

    expected = float(np.quantile(model._score_data(train), 0.9))

    assert model._threshold == pytest.approx(expected, rel=1e-5)


def test_scoring_test_data_never_moves_the_threshold():
    model = UCAD(_config())
    model.fit(_images(6, 0.2, seed=61))
    before = model._threshold
    before_resolved = model._resolve_threshold()

    model.predict(_images(20, 0.95, seed=62))  # wildly out-of-distribution

    # `_resolve_threshold()` is what actually gates `y_pred`; `config.threshold` overrides
    # `_threshold` there, so a leak that wrote `config.threshold` from test scores would move
    # every prediction while leaving `_threshold` itself untouched and this test green.
    assert model._threshold == before
    assert model._resolve_threshold() == before_resolved


def test_repeated_prediction_is_deterministic():
    model = _two_task_model()
    data = _images(4, 0.5, seed=71)

    first = model.predict(data)
    second = model.predict(data)

    np.testing.assert_array_equal(first.anomaly_scores, second.anomaly_scores)
    np.testing.assert_array_equal(first.y_pred, second.y_pred)


def test_strategy_predict_cannot_be_steered_by_the_concept_id():
    strategy = UCADStrategy(UCAD(_config()))
    strategy.learn(_images(4, 0.05, seed=81), concept_id="dark")
    strategy.learn(_images(4, 0.85, seed=82), concept_id="bright")
    data = _images(4, 0.05, seed=83)

    baseline = strategy.predict(data, concept_id="dark").anomaly_scores
    # A scorer that returns a constant would trivially pass every comparison below; rule it out.
    assert len(np.unique(baseline)) > 1, "fixture is degenerate: baseline scores must vary"

    for wrong in ("bright", "does-not-exist", None):
        np.testing.assert_array_equal(strategy.predict(data, concept_id=wrong).anomaly_scores, baseline)

    # Every predict() above left `_current_concept_id` pinned to "bright" (the last-learned
    # concept), so a `_route` that consulted `self._current_concept_id` instead of the learned
    # keys could still have passed. Move the model's own current-concept state mid-stream and
    # check the predicted concept_id argument still cannot move the scores. This additionally
    # assumes the "dark" and "bright" task memories don't coincidentally score `data` identically
    # -- if they did, a `_route` that leaked `_current_concept_id` into task selection would
    # still land on a different task yet report the same score, and this check would not catch it.
    strategy._model.set_current_concept("dark")
    np.testing.assert_array_equal(strategy.predict(data, concept_id="dark").anomaly_scores, baseline)

    strategy._model.set_current_concept("does-not-exist")
    np.testing.assert_array_equal(strategy.predict(data, concept_id="dark").anomaly_scores, baseline)


def test_the_model_holds_no_reference_to_evaluated_data_after_predict():
    model = _two_task_model()
    probe = _images(5, 0.42, seed=91)
    # 5 does not collide with this model's key_size/knowledge_size (3) or either task's fit image
    # count (4), so a leading dimension of 5 anywhere in the model's state can only have come
    # from the probe -- there is no legitimate array this model could hold with that shape.
    # config.batch_size is set to exactly len(probe) below so the whole probe arrives in a single
    # batch; a single batch preserves the leading dimension, so anything retained from it keeps
    # that leading dimension even after the preprocessor resizes/reshapes it.
    model.config.batch_size = len(probe)

    model.predict(probe)

    for name, value in _candidate_state_values(vars(model)):
        if not (isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim > 0 and value.shape[0] == len(probe)):
            continue
        # The leading-dimension match alone is already damning given the collision-free probe
        # size above, but where the full shape also matches the raw probe, confirm the content
        # actually is the probe -- so the failure message can say "retained the evaluated batch"
        # and mean it literally, rather than resting on a shape coincidence.
        as_array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        if as_array.shape == probe.shape and np.array_equal(as_array, probe):
            raise AssertionError(f"UCAD retained the evaluated batch verbatim in '{name}'")
        raise AssertionError(
            f"UCAD retained data shaped like the evaluated batch in '{name}' (leading dim == {len(probe)})"
        )
    assert model._cached_image_scores is None
