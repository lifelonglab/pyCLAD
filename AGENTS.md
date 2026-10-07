# AGENTS.md

pyCLAD is a research library for **continual anomaly detection**. Researchers use it to implement their
methods and datasets and to compare them under one evaluation protocol. Two things matter more than
anything else here:

1. **A number pyCLAD reports must be trustworthy and repeatable.** A subtle evaluation bug is worse than
   a crash, because it ends up in someone's paper.
2. **Adding a method or a dataset should stay easy.** A contributor should subclass one class and be done,
   without touching scenarios, callbacks or metrics.

When these guidelines conflict with speed, choose correctness. For trivial tasks, use judgment.

## 1. How to work

**Think before coding.** State your assumptions. If a request has several readings (which scenario? which
strategy contract? supervised or not?), name them instead of picking one silently. If something is
unclear, stop and ask.

**Simplicity first.** Write the minimum code that solves the problem. No speculative options, no
abstraction for a single use, no handling of cases that cannot happen. A new method should read like the
existing ones next to it or simpler.

**Surgical changes.** Every changed line should trace to the request. Do not reformat, rename or
"improve" neighbouring code without confirming with the user first. 
Remove only what your own change made unused; mention other dead code instead of deleting it.

**Verify, then claim.** Turn the task into a check you can run: a bug gets a test that reproduces it, a
new component gets a test that runs it through a scenario. Run the tests and linters before saying the
work is done, and report failures as they are. If the run may take a lot of time (due to, for example, long scenario),
ask first or identify a quicker run.

## 2. Research integrity rules

These are the rules an agent is most likely to break without noticing. Do not trade them for a better
score or a simpler diff.

- **No test data in training.** `fit()` / `learn()` see training data only. Thresholds, normalisation
  statistics, early stopping, epoch or checkpoint selection and hyperparameters must come from training
  data or a split of it (for example, validation).
- **Respect what the scenario hides.** A concept-agnostic strategy gets no concept ids or boundaries, a
  concept-incremental one gets boundaries only, a concept-aware one gets both. Never recover hidden
  information through a side channel (reading the dataset, counting calls, peeking at concept names).
- **One label convention.** `0` is normal, `1` is an anomaly. Higher `anomaly_scores` mean more
  anomalous. Convert at the boundary when wrapping a library that does it differently.
- **Undefined is `NaN`, never a made-up number.** A base metric that cannot be computed returns `NaN`.
  Continual metrics average through `mean_or_nan`, so `NaN` propagates. Do not skip cells, substitute
  0 or 0.5, or catch the error to keep a run going. See "Undefined values" in `docs/metrics.md`.
- **Randomness goes through `pyclad.seed`.** Components take a `seed` or `rng` argument and fall back to
  `new_generator()`. Never call a bare `np.random.default_rng()` or seed globals inside a component.
  Example scripts call `set_seed(...)` once and pass the returned object to the output writer.
- **Everything that shapes a result is reported.** Hyperparameters, buffer sizes, backbone names and
  seeds go into `additional_info()`, so the JSON output is enough to tell two runs apart.
- **Ports of published methods are honest about differences.** Cite the paper in the class docstring.
  List every departure from the paper or reference code in the docs with the reason. If your numbers
  cannot be compared to the paper's (for example because the reference evaluation leaks), say so. Never
  tune an implementation toward published numbers by weakening the rules above.
- **Do not change what an existing metric, scenario or dataset means without confirming with the human first.**
  If the current behaviour looks wrong, raise it; a fix needs a test and an
  explicit decision, and a different definition needs a new class with a new name.

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

The flow: a **Scenario** walks the train concepts of a **Dataset**, calls `strategy.learn(...)`, then
`strategy.predict(...)` on every test concept, and fires **Callbacks** around each step. A **Strategy**
decides how a **Model** is updated. `ConceptMetricCallback` fills a matrix
`[learned concept][evaluated concept]` from a base metric, and continual metrics summarize that matrix.
`JsonOutputWriter` saves the `info()` of every `InfoProvider` it is given.

## 4. Adding things

Read the closest existing implementation first and follow it. Every addition needs: the class, a test
under the mirrored path in `tests/`, an example script in `examples/`, and potential update in the docs.

## 5. Conventions that tests enforce

- **Optional dependencies stay optional.** The core packages (`pyclad.callbacks`, `pyclad.data`,
  `pyclad.metrics`, `pyclad.scenarios`) must import without `torch`, `pytorch_lightning` or `datasets`
  being loaded. Import heavy or optional packages lazily. A new third-party dependency goes into the
  right extra in `pyproject.toml`, and adding one to the core `dependencies` needs the maintainers'
  agreement. Checked by `tests/test_public_api.py`.
- **`__all__` is sorted** and lists only names the package defines.
- **Pointers to docs name a section**: write `"Data leakage" in docs/vision.md`, not just
  `docs/vision.md`, in source and examples. Checked by `tests/vision/test_docs_references.py`.
- Python 3.10+. Line length 120. Docstrings use Sphinx style (`:param x:`, `:return:`) and explain what
  a reader cannot see from the code.
- Tests: plain `pytest` functions, small synthetic data, a fixed seed. Run expensive work (training, a
  scenario run) once in a `scope="module"` or `scope="class"` fixture and assert on its result in
  several tests. Anything that downloads data or runs for long is `@pytest.mark.longrun`.

## 6. Commands

```bash
pip install -e ".[dev]"            # the dev tools plus the torch and vision extras
pytest tests                       # under a minute; longrun tests are skipped
pytest -m longrun --longrun        # downloads datasets; only when you touched one
black src && isort src && flake8 src
pip install -r docs/requirements.txt && mkdocs serve   # preview the docs
```

## 7. Before you say it is done

- [ ] `pytest tests` passes, and new code has tests that would fail without it.
- [ ] `black --check src`, `isort --check src` and `flake8 src` are clean for the files you touched.
- [ ] No rule from section 2 is bent. If one had to be, it is stated plainly in your summary.
- [ ] A new component has an example script and a docs entry, and reports its configuration through
      `additional_info()`.
- [ ] The diff contains only what the task needed.
- [ ] Your summary says what you verified, what you did not, and any assumption you made.

Commits are small and focused, with an imperative subject line (`Add reservoir replay buffer`). Do not
commit `output.json`, `lightning_logs/`. If there are too many changes in the pull request, say so.
