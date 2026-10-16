# Adding a dataset

A dataset is a list of train concepts and a list of test concepts. For a one-off experiment, building a
`ConceptsDataset` directly is enough (see [your first experiment](../getting_started.md)). A dataset class is worth
it when others will load the same data.

## A dataset class

Subclass `ConceptsDataset` and build the concepts in `__init__`. `read_concepts_from_df` turns a DataFrame with
`concept_id`, `concept_name` and `label` columns into concepts, in `concept_id` order:

```python
--8<-- "docs/snippets/guides/custom_dataset.py:dataset"
```

Things to decide and document:

- **The order of the train concepts is the continual order.** If the data has a natural order, keep it; otherwise
  offer named orderings and put the chosen one in the dataset's `name`.
- **Anomalies in training data.** Unsupervised strategies treat all training data as normal. Either remove the
  anomalies or keep them deliberately, and report which (here, through `additional_info()`).
- **Test concepts** carry labels and share names with the train concepts they belong to. Every test concept should
  hold both classes, or ROC-AUC is undefined for it.

## A downloadable benchmark

pyCLAD's built-in benchmarks are hosted on Hugging Face and downloaded on first use. A benchmark in the tabular CAD
format (a CSV with `task_id`, `task_name`, `task_split` and `label` columns, plus an `orderings.json`) needs only two
attributes:

```python
--8<-- "docs/snippets/guides/custom_dataset.py:hf"
```

For another format, follow `UnswDataset`: take a `cache_dir` argument, and import `datasets` inside `__init__`
rather than at the top of the module, so that importing pyCLAD stays fast.

## Checklist

- [ ] The concept order is fixed and documented; train and test concepts share names.
- [ ] Labels follow `0` = normal, `1` = anomaly; convert when the source uses another convention.
- [ ] The class docstring cites the source of the data.
- [ ] `additional_info()` reports every choice that changes the data, such as the ordering or a preprocessing option.
- [ ] To contribute it: a test that downloads it, marked `@pytest.mark.longrun`, an example in `examples/`, and an
      entry in [Datasets](../datasets.md).
