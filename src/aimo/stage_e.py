"""Stage-E native representation.

각 macro stage g에 대해

    B_g ∈ R^[H × r], B_g^T B_g = I          (QR로 만든 orthonormal basis)
    L_g ∈ R^[r × r]                          (양의 대각을 가진 lower-triangular, 항상 invertible)
    E_g = L_g B_g^T
    z_state[g,t] = E_g s_out[g,t] + a_g
    z_mixer[g,t] = E_g M_g[t]
    z_ffn[g,t]   = E_g F_g[t]

같은 E_g와 offset을 macro 입구와 출구에 쓰므로 `z_out - z_in = z_mixer + z_ffn`이 성립합니다.
서로 다른 stage encoder의 latent 차이(`z_state[g+1] - z_state[g]`)는 이 식과 다릅니다.

decoder는 모든 stage가 공유하는 linear map 하나입니다.

    predicted_logits[g,t] = D z_state[g,t] + b

decoder에는 원문, token ID, 별도 attention network가 들어가지 않습니다. projector는
`B_g B_g^T`이고, 일반 E_g에 대해 `E_g^T E_g`를 projector로 쓰지 않습니다. native lift는
pseudoinverse `E_g^+ = B_g L_g^{-1}`로 정의합니다 (`E_g E_g^+ = I`, `E_g^+ E_g = B_g B_g^T`).

encoder는 stage마다 다르지만 모든 문제에 같은 frozen encoder를 씁니다. 문제별 refitting이나
test-time adaptation을 하지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

STAGE_E_SCHEMA = "aimo-stage-e-v1"
STAGE_E_TENSORS = "stage_e.pt"
STAGE_E_META = "stage_e.json"

# 연구 대조군. 이름은 보고서와 checkpoint에 그대로 남깁니다.
CONTROL_STAGE_E = "stage_e"
CONTROL_RANDOM = "random_r_trained_decoder"
CONTROL_PCA = "pca_r_trained_decoder"
CONTROL_REDUCED_RANK = "output_compression_reduced_rank"
CONTROLS_WITH_FIXED_BASIS = (CONTROL_RANDOM, CONTROL_PCA, CONTROL_REDUCED_RANK)


def orthonormal(raw: Tensor) -> Tensor:
    """QR로 column-orthonormal basis를 만들고 부호를 고정합니다 ([H, r])."""
    q, r = torch.linalg.qr(raw, mode="reduced")
    sign = torch.sign(torch.diagonal(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign


class StageE(nn.Module):
    """G개 stage encoder + 공유 linear decoder."""

    def __init__(
        self,
        n_macro: int,
        hidden_size: int,
        rank: int,
        vocab_size: int,
        *,
        seed: int = 0,
        init_basis: Tensor | None = None,
        freeze_basis: bool = False,
    ) -> None:
        super().__init__()
        if not 1 <= rank <= hidden_size:
            raise ValueError(f"rank must be in [1, H={hidden_size}], got {rank}")
        self.n_macro, self.hidden_size, self.rank, self.vocab_size = (
            n_macro, hidden_size, rank, vocab_size)
        generator = torch.Generator().manual_seed(seed)
        if init_basis is None:
            init_basis = torch.randn(n_macro, hidden_size, rank, generator=generator)
        init_basis = torch.stack([orthonormal(b) for b in init_basis])
        self.basis_raw = nn.Parameter(init_basis.clone(), requires_grad=not freeze_basis)
        self.lower_raw = nn.Parameter(torch.zeros(n_macro, rank, rank))
        self.offset = nn.Parameter(torch.zeros(n_macro, rank))
        self.decoder = nn.Parameter(
            torch.randn(vocab_size, rank, generator=generator) / rank**0.5
        )
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        self.freeze_basis = freeze_basis

    # ------------------------------------------------------------------ encoder
    def basis(self) -> Tensor:
        """B [G, H, r], column orthonormal."""
        return torch.stack([orthonormal(b) for b in self.basis_raw])

    def lower(self) -> Tensor:
        """L [G, r, r]: strict lower + exp(diag). 대각이 양수이므로 항상 invertible입니다."""
        strict = torch.tril(self.lower_raw, diagonal=-1)
        diag = torch.diagonal(self.lower_raw, dim1=-2, dim2=-1).exp()
        return strict + torch.diag_embed(diag)

    def encoder_matrix(self) -> Tensor:
        """E [G, r, H] = L B^T."""
        return self.lower() @ self.basis().transpose(1, 2)

    def encode_state(self, states: Tensor) -> Tensor:
        """states [..., G, T, H] (macro 출구) -> z_state [..., G, T, r]."""
        e = self.encoder_matrix()
        return torch.einsum("...gth,grh->...gtr", states, e) + self.offset[:, None, :]

    def encode_update(self, updates: Tensor) -> Tensor:
        """updates [..., G, T, H] -> [..., G, T, r]. offset은 더하지 않습니다."""
        return torch.einsum("...gth,grh->...gtr", updates, self.encoder_matrix())

    def encode_page(self, state: Tensor, mixer: Tensor, ffn: Tensor) -> dict[str, Tensor]:
        """state [..., G+1, T, H]에서 입구/출구 latent와 update latent를 만듭니다."""
        z_in = self.encode_state(state[..., :-1, :, :])
        z_out = self.encode_state(state[..., 1:, :, :])
        return {
            "z_in": z_in,
            "z_state": z_out,
            "z_mixer": self.encode_update(mixer),
            "z_ffn": self.encode_update(ffn),
        }

    # ------------------------------------------------------------------ geometry
    def project_out(self, vectors: Tensor, g: int) -> Tensor:
        """(I - B_g B_g^T) v. H×H projector를 만들지 않고 r차원으로만 계산합니다."""
        b = self.basis()[g]
        return vectors - (vectors @ b) @ b.T

    def captured_energy(self, vectors: Tensor, g: int) -> tuple[Tensor, Tensor]:
        """(||v B_g||², ||v||²). B가 orthonormal이므로 ||v - vBB^T||² = ||v||² - ||vB||²."""
        b = self.basis()[g]
        return (vectors @ b).pow(2).sum(), vectors.pow(2).sum()

    def pseudo_inverse(self) -> Tensor:
        """E^+ [G, H, r] = B L^{-1}."""
        eye = torch.eye(self.rank).expand(self.n_macro, -1, -1)
        l_inv = torch.linalg.solve_triangular(self.lower(), eye, upper=False)
        return self.basis() @ l_inv

    def lift(self, dz: Tensor, g: int) -> Tensor:
        """latent 변화 dz [..., r]를 native state 변화 [..., H]로 올립니다 (E_g^+ dz)."""
        return dz @ self.pseudo_inverse()[g].T

    # ------------------------------------------------------------------ decoder
    def decode_logits(self, z: Tensor, start: int = 0, end: int | None = None) -> Tensor:
        """z [..., r] -> vocab slice logits. 전체 vocab은 호출자가 chunk로 나눠 부릅니다."""
        end = self.vocab_size if end is None else end
        return z @ self.decoder[start:end].T + self.decoder_bias[start:end]

    def folded_decoder(self, sketch: Tensor) -> tuple[Tensor, Tensor]:
        """(R^T D [q, r], R^T b [q]). submission용 student response sketch."""
        return sketch.T @ self.decoder, sketch.T @ self.decoder_bias

    def parameter_report(self) -> dict:
        g, h, r, v = self.n_macro, self.hidden_size, self.rank, self.vocab_size
        return {
            "basis": g * h * r,
            "lower": g * r * r,
            "offset": g * r,
            "decoder": v * r,
            "decoder_bias": v,
            "total": g * h * r + g * r * r + g * r + v * r + v,
        }


@dataclass
class StageECheckpointMeta:
    """Stage-E checkpoint의 provenance. legacy checkpoint와 schema가 다릅니다."""

    control: str
    rank: int
    n_macro: int
    hidden_size: int
    vocab_size: int
    sensitivity_lambda: float
    npr_topk: int
    sketch: dict
    native: dict  # NativeProvenance.as_dict()
    dataset: dict
    selection: dict = field(default_factory=dict)
    prompt_policy: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    schema: str = STAGE_E_SCHEMA


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_stage_e(model: StageE, meta: StageECheckpointMeta, directory: str | Path) -> Path:
    """frozen encoder를 저장합니다. B/L/E는 계산된 값으로도 함께 남겨 재현을 쉽게 합니다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        tensors = {name: value.detach().cpu().contiguous()
                   for name, value in model.state_dict().items()}
        tensors["frozen/basis"] = model.basis().cpu().contiguous()
        tensors["frozen/lower"] = model.lower().cpu().contiguous()
        tensors["frozen/encoder"] = model.encoder_matrix().cpu().contiguous()
    tensor_path = directory / STAGE_E_TENSORS
    torch.save(tensors, tensor_path)
    payload = asdict(meta)
    payload["tensor_sha256"] = _sha256(tensor_path)
    payload["config_hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "tensor_sha256"},
                   sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    path = directory / STAGE_E_META
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def load_stage_e(directory: str | Path) -> tuple[StageE, dict]:
    """schema와 tensor hash를 확인하고 `weights_only=True`로 읽습니다.

    legacy `aimo-checkpoint-v3`(Looped), `aimo-lrt-v1`(LRT) checkpoint는 여기로 load되지
    않습니다.
    """
    from .native_page import SchemaIncompatible

    directory = Path(directory)
    meta_path = directory / STAGE_E_META
    if not meta_path.exists():
        raise SchemaIncompatible(
            f"SCHEMA_INCOMPATIBLE: {directory} has no {STAGE_E_META}; legacy Looped / LRT "
            "checkpoints cannot be loaded as a Stage-E encoder"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != STAGE_E_SCHEMA:
        raise SchemaIncompatible(f"checkpoint schema {meta.get('schema')!r} != {STAGE_E_SCHEMA!r}")
    tensor_path = directory / STAGE_E_TENSORS
    if _sha256(tensor_path) != meta["tensor_sha256"]:
        raise SchemaIncompatible("Stage-E tensor file does not match the recorded sha256")
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    model = StageE(meta["n_macro"], meta["hidden_size"], meta["rank"], meta["vocab_size"],
                   freeze_basis=meta["control"] in CONTROLS_WITH_FIXED_BASIS)
    model.load_state_dict({k: v for k, v in tensors.items() if not k.startswith("frozen/")})
    for param in model.parameters():
        param.requires_grad_(False)
    return model, meta


def pca_basis(states: Tensor, mask: Tensor, rank: int) -> Tensor:
    """train state [N, G, T, H]의 stage별 PCA basis [G, H, r]. train split만 넣습니다."""
    out = []
    for g in range(states.shape[1]):
        x = states[:, g][mask]  # [n_tokens, H]
        x = x - x.mean(0, keepdim=True)
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        out.append(vh[:rank].T)
    return torch.stack(out)


def random_basis(n_macro: int, hidden: int, rank: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed + 7919)
    return torch.stack(
        [orthonormal(torch.randn(hidden, rank, generator=generator)) for _ in range(n_macro)]
    )


def output_compression_basis(head_weight: Tensor, n_macro: int, rank: int) -> Tensor:
    """output compression / reduced-rank control.

    native head (sketch 좌표의 `R^T W_lm`, [q, H])가 가장 강하게 읽는 hidden 방향 r개를 모든
    stage에 씁니다. 출력만 압축하는 basis가 Stage-E보다 fidelity / gradient capture를 얼마나
    설명하는지 비교하기 위한 대조군입니다.
    """
    _, _, vh = torch.linalg.svd(head_weight.float(), full_matrices=False)
    return vh[:rank].T.unsqueeze(0).expand(n_macro, -1, -1).clone()
