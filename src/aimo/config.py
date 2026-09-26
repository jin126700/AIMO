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
from typing import Any, ClassVar

import yaml

# flow-only (legacy/auxiliary) 이름
FLOW_MODEL_NAMES = ("persistence", "linear", "m0", "loop1", "loop4", "untied4")
# behavior 또는 joint 학습이 가능한 이름. 같은 LoopedCore를 쓰고 head만 다릅니다.
BEHAVIOR_MODEL_NAMES = (
    "constant",
    "raw_change",
    "behavior",
    "behavior_loop1",
    "behavior_loop4",
    "behavior_untied4",
    "behavior_m0",
    "joint",
    "joint_loop1",
    "joint_loop4",
    "joint_untied4",
    "joint_m0",
)
MODEL_NAMES = FLOW_MODEL_NAMES + BEHAVIOR_MODEL_NAMES

# task: 어떤 view를 학습하는지. parameter 구조는 같습니다.
TASKS = ("flow", "behavior", "joint")

# task별로 정의되는 validation 지표. checkpoint 선택 지표는 여기서 골라야 합니다.
FLOW_SELECT_METRICS = ("total", "flow_total", "L_next", "L_within", "L_roll")
BEHAVIOR_SELECT_METRICS = (
    "total",
    "behavior_total",
    "L_robust",
    "L_pair_drop",
    "L_max_drop",
)


def allowed_select_metrics(task: str) -> tuple[str, ...]:
    if task == "flow":
        return FLOW_SELECT_METRICS
    if task == "behavior":
        return BEHAVIOR_SELECT_METRICS
    return tuple(dict.fromkeys(BEHAVIOR_SELECT_METRICS + FLOW_SELECT_METRICS))


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
    # ---- synthetic behavior label (v2) ----
    # toy label은 아래의 명시적 생성 규칙으로 만든 synthetic 값이며 실제 LLM robustness가
    # 아닙니다. delta를 고정 fragility 방향에 투영한 값에서 drop을 만듭니다.
    behavior_labels: bool = True
    planned_trials: int = 8
    fragility_seed: int = 4242
    align_strength: float = 1.0
    proj_gain: float = 0.9
    robust_threshold: float = 0.25
    unresolved_fraction: float = 0.15


@dataclass
class RobustPolicyConfig:
    """연구용 binary robust labeling. 정의가 없으면 robust label은 null로 남습니다."""

    enabled: bool = False
    definition_id: str | None = None
    source: str | None = None


@dataclass
class DataConfig:
    source: str = "synthetic"  # synthetic | pages | deepmath
    page_dir: str | None = None
    label_path: str | None = None  # build-labels 산출물 (pages source와 함께 씁니다)
    pair_path: str | None = None  # import-pairs 산출물
    outcome_path: str | None = None  # collect-outcomes 산출물
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    subset_fraction: float = 1.0  # original-group 단위 nested subset (0.25 / 0.5 / 1.0)
    scale_floor: float = 1e-6  # 이 아래의 scale은 inactive로 표시합니다.
    robust_policy: RobustPolicyConfig = field(default_factory=RobustPolicyConfig)


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
    # task: flow(legacy/auxiliary) | behavior | joint(primary)
    task: str = "flow"
    lr: float = 3e-4
    weight_decay: float = 1e-3
    batch_originals: int = 8  # effective batch
    microbatch_originals: int = 2  # gradient accumulation 단위
    grad_clip: float = 1.0
    max_epochs: int = 100
    patience: int = 15
    cuts_per_pair: int = 2
    horizons: tuple[int, ...] = (2, 4)
    # flow auxiliary
    w_next: float = 1.0
    w_within: float = 1.0
    w_roll: float = 0.25
    flow_weight: float = 0.1  # L = L_behavior + flow_weight * L_flow
    # behavior supervision
    w_robust: float = 1.0
    w_pair_drop: float = 1.0
    w_max_drop: float = 1.0
    # pair drop과 max drop이 같은 counts에서 나오면 이중 감독이므로 기본은 False입니다.
    use_max_drop: bool = False
    # checkpoint 선택 지표. 시작 전에 config로 고정합니다 ("auto"는 task에서 결정).
    select_metric: str = "auto"

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"train.task must be one of {TASKS}, got {self.task!r}")
        allowed = allowed_select_metrics(self.task)
        if self.select_metric != "auto" and self.select_metric not in allowed:
            # task에서 정의되지 않는 지표를 고르면 best checkpoint가 생기지 않으므로 막습니다.
            raise ValueError(
                f"train.select_metric={self.select_metric!r} is never defined for "
                f"train.task={self.task!r}; choose one of {allowed}"
            )

    def resolved_select_metric(self) -> str:
        if self.select_metric != "auto":
            return self.select_metric
        return "total" if self.task == "flow" else "behavior_total"


@dataclass
class EvalConfig:
    horizons: tuple[int, ...] = (2, 4)
    bootstrap_samples: int = 200
    support_swap: bool = True
    behavior: bool = True  # pair drop / robust / panel 지표를 함께 평가합니다.


@dataclass
class ThinkingProfileConfig:
    """어려운 문제용 research thinking profile (v2).

    legacy non-thinking/256-token 설정(`ScreeningConfig`)을 조용히 재사용하지 않습니다.
    이 profile은 Qwen thinking 기반 **연구용** 설정이며 공식 AIMO와 동일하다고 쓰지
    않습니다. 아래 None 값은 서버 calibration manifest에서 확정·freeze해야 하는
    항목(needs_calibration)입니다.
    """

    model_id: str = "Qwen/Qwen3-4B"
    enable_thinking: bool = True
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    # 아래는 calibration에서 확정합니다. 확정 전에는 실제 GPU full run을 막습니다.
    max_new_tokens: int | None = None
    samples_per_prompt: int | None = None
    max_total_context: int | None = None  # prompt tokens + generated tokens 한도
    numerical_backend: str | None = None
    scorer_id: str | None = None
    scorer_version: str | None = None
    rope_scaling: str | None = None  # 변경하면 protocol hash가 달라집니다.
    final_cap_failure_policy: str | None = None  # 최종 cap에서의 실패 점수 정의

    CALIBRATION_FIELDS: ClassVar[tuple[str, ...]] = (
        "max_new_tokens",
        "samples_per_prompt",
        "max_total_context",
        "numerical_backend",
        "scorer_id",
        "scorer_version",
    )

    def needs_calibration(self) -> list[str]:
        """아직 확정되지 않은 항목 목록. 비어 있지 않으면 full run을 막습니다."""
        return [name for name in self.CALIBRATION_FIELDS if getattr(self, name) is None]

    def protocol_hash(self) -> str:
        """실행 환경·policy 변경을 구분하는 hash."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class ScreeningConfig:
    """legacy 초기 screening 설정 (non-thinking, 256 token).

    구현·저난도 대조용으로 보존합니다. 새 DeepMath 경로에서는 ThinkingProfileConfig를
    씁니다. 연구용이며 공식 AIMO 평가 정책이 아닙니다.
    """

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
class DeepMathConfig:
    """DeepMath primary 데이터 경로 설정."""

    dataset_id: str = "zwhe99/DeepMath-103K"
    revision: str | None = None  # pinned snapshot revision. 없으면 SERVER_PENDING.
    local_path: str | None = None  # local JSONL/parquet snapshot
    max_candidate_originals: int = 300  # 후보 수이지 확보된 labeled pair 수가 아닙니다.
    require_topic: bool = True
    allowed_topics: tuple[str, ...] = (
        "algebra",
        "number theory",
        "combinatorics",
        "discrete",
        "geometry",
    )


@dataclass
class ServerConfig:
    repo_root: str = "/data1/HKM/AIMO"
    input_root: str = "/data1/HKM/data"
    output_root: str = "/data1/HKM/result/AIMO/HKM/aimo_v2"
    model_id: str = "Qwen/Qwen3-4B"
    tiny_config: bool = True  # CPU 검증에서는 random-init tiny config만 사용합니다.
    gpu_block_minutes: int = 175
    gpu_stop_minutes: int = 180
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    thinking: ThinkingProfileConfig = field(default_factory=ThinkingProfileConfig)
    mathgap: MathGapConfig = field(default_factory=MathGapConfig)
    deepmath: DeepMathConfig = field(default_factory=DeepMathConfig)
    # 기본 GPU 예산 3시간을 run config로 관리합니다.
    gpu_budget_minutes: int = 180


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
