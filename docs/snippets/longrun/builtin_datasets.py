# --8<-- [start:unsw]
from pyclad.data.datasets import UnswDataset

unsw = UnswDataset(dataset_type="random_anomalies", cache_dir=None)  # None: the default Hugging Face cache
print(unsw.name(), [concept.name for concept in unsw.train_concepts()])
# --8<-- [end:unsw]

# --8<-- [start:cad]
from pyclad.data.datasets import CadCicids2017Dataset

print(CadCicids2017Dataset.available_orderings())  # e.g. ['curriculum_asc', 'curriculum_desc', ...]
cad = CadCicids2017Dataset(ordering="curriculum_asc")
# --8<-- [end:cad]
print(cad.name(), len(cad.train_concepts()))
