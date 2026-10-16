# requires: torch, torchvision
# needs local data: the BTech dataset, downloaded and transformed as described in "Datasets" in docs/guides/vision.md
# --8<-- [start:read]
import pathlib

from pyclad.vision.data.readers.vision_reader import read_vision_dataset

dataset = read_vision_dataset(
    root=pathlib.Path("../../resources/vision/BTech_Dataset_transformed"),  # dataset folder
    benchmark="btech",  # naming convention: btech | mvtec | mpdd | visa | dagm
    resize_to=(256, 256),  # all images resized to this H×W (use 224 for PaSTe)
    data_mode="numpy",  # "numpy" = load into memory, "paths" = load lazily
    color_mode="rgb",  # "rgb" or "grayscale"
    # max_train_samples_per_category=150,   # uncomment → fewer images per category
    # max_test_samples_per_category=150,
)
# --8<-- [end:read]
