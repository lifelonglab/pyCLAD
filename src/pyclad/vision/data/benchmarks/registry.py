import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.vision.data._paths import (
    DEFAULT_VISION_DATASET_REGISTRY_PATH,
    DEFAULT_VISION_DATASETS_ROOT,
)
from pyclad.vision.data.benchmarks.readers import (
    PREDEFINED_BENCHMARK_ALIASES,
    VisionBenchmarkReader,
    VisionBenchmarkSpec,
    build_vision_benchmark_reader,
    index_vision_benchmark,
    read_vision_benchmark_dataset,
)

VISION_BENCHMARK_ENV_VARS = {
    "btech": "PYCLAD_BTECH_ROOT",
    "dagm": "PYCLAD_DAGM_ROOT",
    "mpdd": "PYCLAD_MPDD_ROOT",
    "mvtec": "PYCLAD_MVTEC_ROOT",
    "visa": "PYCLAD_VISA_ROOT",
}

VISION_BENCHMARK_DIRECTORY_CANDIDATES = {
    "btech": ("BTech_Dataset_transformed", "btech_dataset_transformed", "btech"),
    "dagm": ("DAGM_KaggleUpload", "dagm_kaggleupload", "dagm"),
    "mpdd": ("MPDD", "mpdd"),
    "mvtec": ("mvtec_ad", "mvtec", "mvtec_anomaly_detection"),
    "visa": ("visa", "VisA"),
}

VISION_BENCHMARK_SHARED_ROOT_ENV = "PYCLAD_VISION_DATASETS_ROOT"
VISION_BENCHMARK_REGISTRY_ENV = "PYCLAD_VISION_DATASETS_FILE"
VISION_BENCHMARK_DOWNLOAD_URLS = {
    "btech": "https://github.com/pankajmishra000/VT-ADL",
    "dagm": "https://zenodo.org/records/12750201",
    "mpdd": "https://github.com/stepanje/MPDD",
    "mvtec": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
    "visa": "https://github.com/amazon-science/spot-diff",
}


def resolve_vision_dataset_registry_path(registry_path: Optional[Union[str, Path]] = None) -> Path:
    if registry_path is not None:
        return Path(registry_path).expanduser().resolve()

    env_path = os.getenv(VISION_BENCHMARK_REGISTRY_ENV)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return DEFAULT_VISION_DATASET_REGISTRY_PATH


def load_vision_dataset_registry(registry_path: Optional[Union[str, Path]] = None) -> dict[str, str]:
    path = resolve_vision_dataset_registry_path(registry_path)
    if not path.exists():
        return {}

    content = json.loads(path.read_text())
    if not isinstance(content, dict):
        raise ValueError(f"Vision dataset registry must be a JSON object, got {type(content).__name__}")

    registry = {}
    for benchmark_name, root_path in content.items():
        if not isinstance(benchmark_name, str):
            raise ValueError("Vision dataset registry keys must be strings")
        if root_path in (None, ""):
            continue
        if not isinstance(root_path, str):
            raise ValueError(f"Vision dataset registry values must be strings, got {type(root_path).__name__}")
        registry[_normalize_vision_benchmark_name(benchmark_name)] = root_path
    return registry


def write_vision_dataset_registry(
    entries: Mapping[str, Union[str, Path]],
    registry_path: Optional[Union[str, Path]] = None,
) -> Path:
    path = resolve_vision_dataset_registry_path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = {}
    for benchmark_name, root_path in entries.items():
        serialized[_normalize_vision_benchmark_name(benchmark_name)] = str(Path(root_path).expanduser())

    path.write_text(json.dumps(serialized, indent=4, sort_keys=True))
    return path


def resolve_vision_benchmark_root(
    benchmark: Union[str, VisionBenchmarkSpec],
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
) -> Path:
    benchmark_name = _normalize_vision_benchmark_name(benchmark)

    if root is not None:
        resolved_root = Path(root).expanduser().resolve()
        if resolved_root.exists():
            return resolved_root
        _raise_missing_vision_benchmark_root(
            benchmark_name=benchmark_name,
            registry_path=registry_path,
            attempted_paths=[resolved_root],
            explicit_root=resolved_root,
        )

    registry = load_vision_dataset_registry(registry_path)
    if benchmark_name in registry:
        resolved_root = Path(registry[benchmark_name]).expanduser().resolve()
        if resolved_root.exists():
            return resolved_root

    benchmark_env = VISION_BENCHMARK_ENV_VARS.get(benchmark_name)
    if benchmark_env:
        env_root = os.getenv(benchmark_env)
        if env_root:
            resolved_root = Path(env_root).expanduser().resolve()
            if resolved_root.exists():
                return resolved_root

    shared_root = os.getenv(VISION_BENCHMARK_SHARED_ROOT_ENV)
    attempted_paths: list[Path] = []
    for candidate in _benchmark_root_candidates(benchmark_name=benchmark_name, shared_root=shared_root):
        attempted_paths.append(candidate)
        if candidate.exists():
            return candidate

    _raise_missing_vision_benchmark_root(
        benchmark_name=benchmark_name,
        registry_path=registry_path,
        attempted_paths=attempted_paths,
    )


def build_registered_vision_benchmark_reader(
    benchmark: Union[str, VisionBenchmarkSpec],
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
) -> VisionBenchmarkReader:
    resolved_root = resolve_vision_benchmark_root(benchmark=benchmark, root=root, registry_path=registry_path)
    return build_vision_benchmark_reader(root=resolved_root, benchmark=benchmark)


def read_registered_vision_benchmark_dataset(
    benchmark: Union[str, VisionBenchmarkSpec],
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
    dataset_name: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> ConceptsDataset:
    resolved_root = resolve_vision_benchmark_root(benchmark=benchmark, root=root, registry_path=registry_path)
    return read_vision_benchmark_dataset(
        root=resolved_root,
        benchmark=benchmark,
        dataset_name=dataset_name,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def index_registered_vision_benchmark(
    benchmark: Union[str, VisionBenchmarkSpec],
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
    categories: Optional[Sequence[str]] = None,
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
):
    resolved_root = resolve_vision_benchmark_root(benchmark=benchmark, root=root, registry_path=registry_path)
    return index_vision_benchmark(
        root=resolved_root,
        benchmark=benchmark,
        categories=categories,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def _normalize_vision_benchmark_name(benchmark: Union[str, VisionBenchmarkSpec]) -> str:
    if isinstance(benchmark, str):
        key = benchmark.lower()
        return PREDEFINED_BENCHMARK_ALIASES.get(key, key)
    return PREDEFINED_BENCHMARK_ALIASES.get(benchmark.name.lower(), benchmark.name.lower())


def _benchmark_root_candidates(benchmark_name: str, shared_root: Optional[str]) -> list[Path]:
    candidate_paths: list[Path] = []

    if shared_root:
        base_root = Path(shared_root).expanduser().resolve()
        candidate_paths.extend(_dataset_candidates_under_root(base_root, benchmark_name))

    default_root = DEFAULT_VISION_DATASETS_ROOT.expanduser().resolve()
    if not shared_root or default_root != Path(shared_root).expanduser().resolve():
        candidate_paths.extend(_dataset_candidates_under_root(default_root, benchmark_name))

    return candidate_paths


def _dataset_candidates_under_root(base_root: Path, benchmark_name: str) -> list[Path]:
    return [
        base_root / dirname for dirname in VISION_BENCHMARK_DIRECTORY_CANDIDATES.get(benchmark_name, (benchmark_name,))
    ]


def _raise_missing_vision_benchmark_root(
    benchmark_name: str,
    registry_path: Optional[Union[str, Path]],
    attempted_paths: Sequence[Path],
    explicit_root: Optional[Path] = None,
) -> None:
    registry_file = resolve_vision_dataset_registry_path(registry_path)
    benchmark_env = VISION_BENCHMARK_ENV_VARS.get(benchmark_name)
    download_url = VISION_BENCHMARK_DOWNLOAD_URLS.get(benchmark_name)
    default_candidates = _dataset_candidates_under_root(
        DEFAULT_VISION_DATASETS_ROOT.expanduser().resolve(), benchmark_name
    )
    suggested_target = (
        default_candidates[0] if default_candidates else DEFAULT_VISION_DATASETS_ROOT.expanduser().resolve()
    )

    message_lines = [f"Could not resolve local root for vision benchmark '{benchmark_name}'."]
    if explicit_root is not None:
        message_lines.append(f"The provided root does not exist: {explicit_root}")
    if download_url is not None:
        message_lines.append(f"Please download the dataset from: {download_url}")
    message_lines.append(f"Place it in the default folder: {suggested_target}")
    message_lines.append(
        f"Or configure it via root=..., {benchmark_env}, {VISION_BENCHMARK_SHARED_ROOT_ENV}, "
        f"or the registry file {registry_file}."
    )
    if attempted_paths:
        message_lines.append(f"Checked paths: {', '.join(str(p) for p in attempted_paths)}")

    raise FileNotFoundError(" ".join(message_lines))
