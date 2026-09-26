"""YAML config 로딩과 config hash.

Config는 평범한 nested dataclass입니다. 미지의 key는 오류로 처리해 오타가 조용히
무시되지 않게 합니다. config_hash는 checkpoint/run directory의 mismatch 차단에
사용합니다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

MODEL_NAMES = ("persistence", "linear", "m0", "loop1", "loop4", "untied4")


@dataclass
class RunConfig:
    run_id: str = "toy_e0"
    seed: int = 0
    device: str = "cpu"  # 로컬은 cpu 고정. GPU 실행은 CLI의 --execute-gpu가 필요합니다.
    notes: str = ""


@dataclass
class PathsConfig:
    input_root: str = "data"
    output_root: str = "runs"


@dataclass
class SyntheticConfig:
    """E0 synthetic Page 생성 설정. 실제 robust dataset이 아닙니다."""

    hidden_size: int = 32
    n_blocks: int = 6
    n_landmarks: int = 4
    n_originals_train: int = 16
    n_originals_validation: int = 6
    n_originals_test: int = 6
    variants_per_original: int = 3
    identity_fraction: float = 0.1
    invalid_landmark_prob: float = 0.15
    noise_scale: float = 0.005
    # variant delta 세기. sibling 신호가 update 크기에 비해 충분히 커야 학습 신호가 됩니다.
    delta_scale: float = 5.0
    harder_delta_scale: float = 11.0


@dataclass
class DataConfig:
    source: str = "synthetic"  # synthetic | pages
    page_dir: str | None = None
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    subset_fraction: float = 1.0  # original-group 단위 nested subset (0.25 / 0.5 / 1.0)
    scale_floor: float = 1e-6  # 이 아래의 scale은 inactive로 표시합니다.


@dataclass
class ModelConfig:
    name: str = "loop4"
    d_model: int = 128
    n_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    n_loops: int = 4
    tied: bool = True
    use_variant_prefix: bool = True  # M0는 False: pair-specific 정보를 주지 않습니다.

    def __post_init__(self) -> None:
        if self.name not in MODEL_NAMES:
            raise ValueError(f"model.name must be one of {MODEL_NAMES}, got {self.name!r}")


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-3
    batch_originals: int = 8  # effective batch
    microbatch_originals: int = 2  # gradient accumulation 단위
    grad_clip: float = 1.0
    max_epochs: int = 100
    patience: int = 15
    cuts_per_pair: int = 2
    horizons: tuple[int, ...] = (2, 4)
    w_next: float = 1.0
    w_within: float = 1.0
    w_roll: float = 0.25


@dataclass
class EvalConfig:
    horizons: tuple[int, ...] = (2, 4)
    bootstrap_samples: int = 200
    support_swap: bool = True


@dataclass
class ScreeningConfig:
    """초기 screening 설정. 연구용이며 공식 AIMO 평가 정책이 아닙니다."""

    thinking: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    max_new_tokens: int = 256
    slots_per_prompt: int = 4


@dataclass
class MathGapConfig:
    """MathGAP 연결 경로. 서버 운영자가 확인한 dotted path만 채웁니다.

    비어 있으면 adapter가 SERVER_PENDING으로 fail-fast합니다. 확인하지 못한 API를
    추측해 채우지 않습니다.
    """

    revision: str = "SERVER_PENDING"
    generator_path: str | None = None
    renderer_path: str | None = None
    oracle_path: str | None = None


@dataclass
class ServerConfig:
    repo_root: str = "/data1/HKM/AIMO"
    input_root: str = "/data1/HKM/data"
    output_root: str = "/data1/HKM/result/AIMO/HKM/aimo_v1"
    model_id: str = "Qwen/Qwen3-4B"
    tiny_config: bool = True  # CPU 검증에서는 random-init tiny config만 사용합니다.
    gpu_block_minutes: int = 175
    gpu_stop_minutes: int = 180
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    mathgap: MathGapConfig = field(default_factory=MathGapConfig)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.paths.output_root) / self.run.run_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _build(cls: type, payload: Any, path: str) -> Any:
    if not is_dataclass(cls):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"config section {path!r} must be a mapping, got {type(payload).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(f"unknown config key(s) in {path or 'root'}: {sorted(unknown)}")
    kwargs = {}
    for name, value in payload.items():
        spec = known[name]
        child = f"{path}.{name}" if path else name
        if is_dataclass(spec.type) if isinstance(spec.type, type) else False:
            kwargs[name] = _build(spec.type, value, child)
        elif isinstance(value, dict) and is_dataclass(_resolve(cls, name)):
            kwargs[name] = _build(_resolve(cls, name), value, child)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _resolve(cls: type, name: str) -> Any:
    """nested dataclass field의 실제 type을 default_factory로 알아냅니다."""
    for spec in fields(cls):
        if spec.name == name and spec.default_factory is not None:  # type: ignore[misc]
            try:
                return type(spec.default_factory())  # type: ignore[misc]
            except TypeError:
                return None
    return None


def config_from_dict(payload: dict[str, Any]) -> Config:
    return _build(Config, payload or {}, "")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """YAML 하나를 읽고 dotted-key override를 적용합니다."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for dotted, value in (overrides or {}).items():
        _set_dotted(payload, dotted, value)
    return config_from_dict(payload)


def _set_dotted(payload: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = payload
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override inside scalar key {key!r}")
    node[keys[-1]] = value
