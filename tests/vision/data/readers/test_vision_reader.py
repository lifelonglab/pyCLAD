from pathlib import Path

from pyclad.vision.data.readers.vision_reader import build_vision_reader, read_vision_dataset
from pyclad.vision.data.readers.generic_reader import GenericFolderReader
from tests.vision._helpers import write_rgb_image as _write_rgb_image


def test_readers_vision_reader_is_public_entrypoint(tmp_path: Path):
    _write_rgb_image(tmp_path / "bottle" / "train" / "good" / "001.png")
    _write_rgb_image(tmp_path / "bottle" / "test" / "good" / "010.png")

    reader = build_vision_reader(root=tmp_path, name="mydata")

    assert isinstance(reader, GenericFolderReader)

    dataset = read_vision_dataset(root=tmp_path, name="mydata", data_mode="paths")
    assert dataset.train_concepts()[0].name == "bottle"
