# requires: torch, torchvision
# needs local data: the source datasets of the stream, placed as described in "Datasets" in docs/guides/vision.md
from pyclad.vision.data.multi_dataset import MultiDatasetSpec

spec = MultiDatasetSpec.from_json("my_stream.json")

# --8<-- [start:read]
from pyclad.vision.data.multi_dataset import load_inclad_md, read_multi_dataset

roots = {"mvtec": "/data/mvtec_ad", "btech": "/data/BTech_Dataset_transformed"}  # blocks without their own root
dataset = read_multi_dataset(spec, roots=roots, resize_to=(256, 256))

# The published InCLAD-MD benchmark needs a root for each of its five source datasets.
inclad = load_inclad_md(roots={...}, ordering="easy_to_hard")  # or "hard_to_easy"
# --8<-- [end:read]
