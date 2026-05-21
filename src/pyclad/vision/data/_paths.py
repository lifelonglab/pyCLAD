from pathlib import Path

USER_VISION_DATA_DIR = Path.home() / ".pyclad" / "vision"
REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_VISION_DATASET_REGISTRY_PATH = USER_VISION_DATA_DIR / "registry.json"
DEFAULT_VISION_DATASETS_ROOT = REPO_ROOT / "examples" / "resources" / "vision"
DEFAULT_VISION_BENCHMARK_MANIFESTS_DIR = USER_VISION_DATA_DIR / "manifests"
DEFAULT_VISION_BENCHMARK_ORDERINGS_DIR = USER_VISION_DATA_DIR / "orderings"
