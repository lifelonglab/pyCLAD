# AGENTS.md

pyCLAD is a research library for **continual anomaly detection**. Researchers use it to implement their
methods and datasets and to compare them under one evaluation protocol. Two things matter more than
anything else here:

1. **A number pyCLAD reports must be trustworthy and repeatable.** A subtle evaluation bug is worse than
   a crash, because it ends up in someone's paper.
2. **Adding a method or a dataset should stay easy.** A contributor should subclass one class and be done,
   without touching scenarios, callbacks or metrics.

When these guidelines conflict with speed, choose correctness. For trivial tasks, use judgment.
"The user" below is the person you are working for. When a rule says to ask, ask them.

## 1. How to work

**Think before coding.** State your assumptions. If a request has several readings (which scenario? which
strategy contract? supervised or not?), name them and ask instead of picking one silently.

**Simplicity first.** Write the minimum code that solves the problem, with nothing speculative. A new
method should read like the existing ones next to it or simpler.

**Surgical changes.** Every changed line should trace to the request. Do not reformat, rename, "improve"
or delete neighbouring code without confirming with the user first.

**Verify, then claim.** A bug gets a test that reproduces it; a new component gets a test that runs it
through a scenario. Run the tests and linters before saying the work is done. If a run may take long
(for example, a long scenario), ask first or find a quicker one. Your change must add no new failures;
name checks that already failed before it instead of fixing them in the same change. Say in your
summary what you verified, what you did not, and what you assumed.

**Do not leave maintenance traps.** A trap is code that works today and breaks quietly when someone
changes something else. Before finishing, ask what the next person has to know to change your code
safely, and remove the need to know it:

- The same logic or fact in two places that must stay equal. Reuse the existing one; if you cannot,
  say where the copy is.
- Behaviour that depends on something implicit: argument or list order, a string that must match another
  string, a default inherited from a third-party library, global state.
- A failure that stays silent: a broad `except`, a fallback value, a skipped item, a dictionary key that
  overwrites another. Fail loudly instead.
- State that carries over between `learn()` calls without a clear decision whether it resets.
- A rule that only a comment or a docstring enforces. If it matters, a test or a check enforces it.

If a trap already exists where you work, or you cannot avoid adding one, say so in your summary.

## 2. Research integrity rules

These are the rules an agent is most likely to break without noticing. Do not trade them for a better
score or a simpler diff. If one had to be bent, state it plainly in your summary.

- **No test data in training.** `fit()` / `learn()` see training data only. Thresholds, normalisation
  statistics, early stopping, epoch or checkpoint selection and hyperparameters must come from training
  data or a split of it (for example, validation). `predict()` does not change learned state.
- **Respect what the scenario hides.** A concept-agnostic strategy is given no concept ids or boundaries,
  a concept-incremental one is given boundaries only (each `learn()` call is one concept), a
  concept-aware one is given both. A method may infer what it is not given from the data it legitimately
  receives; drift detection and routing a sample to a learned concept are exactly that. It may not
  obtain it any other way: reading the dataset object, concept names, labels, or the order of evaluation.
- **One label convention.** `0` is normal, `1` is an anomaly. Higher `anomaly_scores` mean more
  anomalous. Convert at the boundary when wrapping a library that does it differently.
- **Undefined is `NaN`, never a made-up number.** A base metric that cannot be computed returns `NaN`.
  Continual metrics average through `mean_or_nan`, so `NaN` propagates. Do not skip cells or substitute
  0 or 0.5. The only way to let a run continue past an undefined value is the callbacks'
  `on_undefined="propagate"`, which records it. See "Undefined values" in `docs/metrics.md`.
- **Randomness goes through `pyclad.seed`.** A component takes a `seed` or `rng` argument and, given
  none, uses `new_generator()`. Never create a generator from fresh entropy (a bare
  `np.random.default_rng()`). Only `set_seed` seeds Python's `random` and NumPy globally. A component
  with its own seed may seed torch with it, as `vision/models/utilities/base_model.py` does. New example
  scripts call `set_seed(...)` once and pass the returned object to the output writer; the existing
  examples do not do this yet.
- **Everything that shapes a result is reported.** Hyperparameters, buffer sizes, backbone names and
  seeds go into `additional_info()`, so the JSON output is enough to tell two runs apart.
- **Ports of published methods are honest about differences.** Cite the paper in the class docstring.
  List every departure from the paper or reference code in the docs with the reason. If your numbers
  cannot be compared to the paper's (for example because the reference evaluation leaks), say so. Never
  tune an implementation toward published numbers by weakening the rules above.
- **Do not change what an existing metric, scenario or dataset means without confirming with the user
  first.** If the current behaviour looks wrong, raise it; a fix needs a test and an explicit decision,
  and a different definition needs a new class with a new name.

## 3. Where things live

```
src/pyclad/
  data/          Concept, ConceptsDataset, step schedules (grouping.py), time-series windows
    datasets/    built-in benchmarks, downloaded from Hugging Face on first use
  models/        Model, SupervisedModel, TorchBackbone, PyOD and torch adapters, autoencoders
  strategies/    Strategy contracts; baselines, replay, regularization, vlad
  scenarios/     the loops that drive learn / predict and fire callbacks
  callbacks/     Callback hooks; metric, time, memory and energy evaluation
  metrics/       base/ scores one concept; continual/ summarizes the concept-level matrix
  output/        InfoProvider, PredictionResults, JsonOutputWriter
  vision/        image models, datasets, pixel metrics and strategies (mirrors the layout above)
  seed.py        set_seed, new_generator
tests/           mirrors src/pyclad
examples/        runnable scripts, one per feature
docs/            mkdocs pages, one per component
```

The flow: a **Scenario** walks the train concepts of a **Dataset**; after each `strategy.learn(...)` it
calls `strategy.predict(...)` on every test concept. `ConceptMetricCallback` fills a matrix
`[learned concept][evaluated concept]`, and `JsonOutputWriter` saves the `info()` of every
`InfoProvider` it is given.

## 4. Adding things

Read the closest existing implementation first and follow it. Every addition needs: the class, a test
under the mirrored path in `tests/`, an example script in `examples/`, and an entry in the matching
`docs/*.md` page. Paths below are relative to `src/pyclad/`.

| To add | Subclass | Copy from | Remember |
|---|---|---|---|
| Model | `Model` (`models/model.py`) | `models/adapters/pyod_adapters.py` | `SupervisedModel` is a separate contract for models trained on labels. Torch models implement `TorchBackbone` and are wrapped in `TorchModelAdapter`. Image models subclass `VisionModel`. |
| Strategy | one of the contracts in `strategies/strategy.py`, or `SupervisedStrategy` | `strategies/baselines/naive.py` | Implement only the contracts the method can honestly satisfy. Add a smoke test by subclassing `BaseStrategyTest` in `tests/strategies/smoke_tests/`. |
| Dataset | `ConceptsDataset`, or `TabularCadDataset` for that format | `data/datasets/unsw_dataset.py` | List order is the continual order. Host data on Hugging Face, take a `cache_dir`, import `datasets` inside the function. Cite the source. The download test is `longrun`. |
| Metric | `BaseMetric`, or a base in `metrics/continual/concepts_metric.py` | `metrics/base/roc_auc.py`, `metrics/continual/average_continual.py` | Return `NaN` when undefined. Square-only metrics call `validate_square_matrix`. Test on a hand-computed matrix. |
| Callback | `Callback`, plus `InfoProvider` if it has results | `callbacks/evaluation/time_evaluation.py` | Hooks accept `*args, **kwargs`. Release resources in `after_scenario`. |

Export the new class from its package `__init__` where that package has an `__all__`.

## 5. Running experiments

When the task is to run or script an experiment, the code is only half of it.

- Call `set_seed` at the start and pass what it returns to the output writer. One seed is one sample:
  for a claim that one method beats another, run several seeds and report mean and spread.
- Choose hyperparameters without looking at test concept scores. If you tuned on them, say so next to
  the result.
- Compare methods on the same dataset, concept order, scenario, step schedule and base metric. State
  each of them with the numbers.
- Take numbers from the JSON that `JsonOutputWriter` wrote, never from memory or logs, and keep the file.
- Report every run, including failed ones and `NaN` results. Do not drop or rerun a run because its
  number looks bad.
- Do not edit library code to make an experiment work without telling the user; a result then depends
  on a private version of pyCLAD.

## 6. Conventions

Checked by tests:

- **Optional dependencies stay optional.** The core packages (`pyclad.callbacks`, `pyclad.data`,
  `pyclad.metrics`, `pyclad.scenarios`) must import without `torch`, `pytorch_lightning` or `datasets`
  being loaded. Import heavy or optional packages lazily. A new third-party dependency goes into the
  right extra in `pyproject.toml`; ask the user before adding one to the core `dependencies`. Checked by
  `tests/test_public_api.py`.
- **`__all__` is sorted** and lists only names the package defines.
- **Pointers to docs name a section**: write `"Data leakage" in docs/vision.md`, not just
  `docs/vision.md`, in source and examples. Checked by `tests/vision/test_docs_references.py`.

Not checked, so check them yourself:

- Python 3.10+. Line length 120. Docstrings use Sphinx style (`:param x:`, `:return:`) and explain what
  a reader cannot see from the code.
- Tests use small synthetic data and a fixed seed, as functions or classes like the tests next to them.
  Run expensive work (training, a scenario run) once in a `scope="module"` or `scope="class"` fixture
  and assert on its result in several tests. Anything that downloads data or runs for long is
  `@pytest.mark.longrun`.

## 7. Commands and commits

Use the project's virtual environment (`.venv/` if it exists). Do not install into the system Python.

```bash
pip install -e ".[dev]"            # the dev tools plus the torch and vision extras
pytest tests                       # under a minute; longrun tests are skipped
pytest -m longrun --longrun        # downloads datasets; only when you touched one
black src && isort src && flake8 src
pip install -r docs/requirements.txt && mkdocs serve   # preview the docs
```

Commits are small and focused, with an imperative subject line (`Add reservoir replay buffer`). Do not
commit `output.json` or `lightning_logs/`. If there are too many changes in the pull request, say so.

## 8. Reviewing code

When asked to review a change (yours or a contributor's), look for what could put a wrong number in a
paper before anything else. In this order:

1. **Section 2, rule by rule.** Trace where every array that reaches `fit()` / `learn()` comes from, and
   what `predict()` reads and writes. Leakage rarely looks like leakage: a scaler fitted on the whole
   dataset, a threshold from test scores, early stopping on the test concept, state updated in `predict()`.
2. **Correctness of the method.** For a port, compare against the paper or the reference code and check
   that every difference is listed in the docs. For a metric, recompute one small matrix by hand.
3. **Contracts.** The right base class, the label and score conventions, `additional_info()` complete,
   optional dependencies imported lazily.
4. **Tests.** Would they fail if the implementation were wrong, or do they only check that it runs?
5. **Maintenance traps** (the list in section 1). Search for other callers and for parallel
   implementations (the scenario classes, core and vision callbacks) before judging a change local.
6. **Scope and simplicity.** Changes the task did not need, abstractions with one use.

Report findings ordered by severity, each with the file and line, a concrete case where it goes wrong,
and whether you confirmed it by running something or only read the code. Say what you did not check.
Leave formatting to `black`, `isort` and `flake8`. Do not fix things during a review unless asked.
