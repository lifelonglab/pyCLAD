from pathlib import Path

from PIL import Image

from pyclad.vision.data.readers.vision_reader import build_vision_reader, read_vision_dataset
from pyclad.vision.data.generic_reader import GenericFolderReader


def _write_rgb_image(path: Path, rgb: tuple[int, int, int] = (128, 64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (4, 4), rgb)
    img.save(path)


def test_readers_vision_reader_is_public_entrypoint(tmp_path: Path):
    _write_rgb_image(tmp_path / "bottle" / "train" / "good" / "001.png")
    _write_rgb_image(tmp_path / "bottle" / "test" / "good" / "010.png")

    reader = build_vision_reader(root=tmp_path, name="mydata")

    assert isinstance(reader, GenericFolderReader)

    dataset = read_vision_dataset(root=tmp_path, name="mydata", data_mode="paths")
    assert dataset.train_concepts()[0].name == "bottle"
