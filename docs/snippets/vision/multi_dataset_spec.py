# requires: torch, torchvision
# --8<-- [start:spec]
from pyclad.vision.data.multi_dataset import INCLAD_MD_SPEC, DatasetBlock, MultiDatasetSpec

spec = MultiDatasetSpec(
    name="my-stream",
    blocks=[
        # Block order, then category order within a block, is the continual order.
        DatasetBlock(dataset="mvtec", categories=["bottle", "cable"]),
        DatasetBlock(dataset="visa", categories=["candle"], root="/data/VisA"),  # a block may set its own root
        DatasetBlock(dataset="btech", categories=["01", "02"], alias="btad"),  # alias prefixes the concept names
    ],
)
print(spec.category_order())  # ['mvtec__bottle', 'mvtec__cable', 'visa__candle', 'btad__01', 'btad__02']

spec.to_json("my_stream.json")  # save the stream with the results, so it can be rebuilt exactly
assert MultiDatasetSpec.from_json("my_stream.json") == spec

print(len(INCLAD_MD_SPEC.category_order()))  # the published InCLAD-MD stream: 27 categories
# --8<-- [end:spec]
