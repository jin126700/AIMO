"""Synthetic Page 생성, split, normalization, sibling sampling, batching.

E0 단계에서 쓰는 synthetic Page는 exact residual identity와 nonzero sibling
dynamics만 보장하는 검증용 자료입니다. 실제 robust dataset이 아닙니다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .config import Config, SyntheticConfig
from .page import Page, load_pages, save_pages

SPLITS = ("train", "validation", "known_test", "unseen_perturbation_test", "harder_test")


# --------------------------------------------------------------------------------------
# 자료 구조
# --------------------------------------------------------------------------------------


@dataclass
class OriginalGroup:
    """한 original과 그 sibling variants. split은 group 단위로 나눕니다."""

    original_id: str
    original: Page
    variants: list[Page]


@dataclass
class PageDataset:
    split: str
    groups: list[OriginalGroup]

    def __len__(self) -> int:
        return len(self.groups)

    @property
    def n_blocks(self) -> int:
        return self.groups[0].original.n_blocks

    @property
    def n_landmarks(self) -> int:
        return self.groups[0].original.n_landmarks

    @property
    def hidden_size(self) -> int:
        return self.groups[0].original.hidden_size

    def subset(self, fraction: float) -> PageDataset:
        """original-group 단위의 고정 nested subset (25/50/100%)."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        n_keep = max(1, int(math.ceil(len(self.groups) * fraction)))
        # groups는 결정적으로 정렬되어 있으므로 prefix가 곧 nested subset입니다.
        return PageDataset(split=self.split, groups=self.groups[:n_keep])

    def data_hash(self) -> str:
        payload = [
            [g.original_id] + [v.variant_id for v in g.variants] for g in self.groups
        ]
        blob = json.dumps([self.split, payload], sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class PredictorInput:
    """cut d에서의 predictor 입력. variant future는 여기에 들어오지 않습니다.

    var_state_prefix는 depth 0..d, var_updates_prefix는 depth 0..d-1만 담습니다.
    """

    cut: int
    orig_state: Tensor  # [B, L+1, P, H]
    orig_updates: Tensor  # [B, L, P, 2, H]
    var_state_prefix: Tensor  # [B, d+1, P, H]
    var_updates_prefix: Tensor  # [B, d, P, 2, H]
    valid: Tensor  # [B, P] bool, original AND variant 공통 landmark
    relative_positions: Tensor  # [B, P] float32
    token_offsets: Tensor  # [B, P] int64

    @property
    def batch_size(self) -> int:
        return int(self.orig_state.shape[0])

    @property
    def n_blocks(self) -> int:
        return int(self.orig_updates.shape[1])


@dataclass
class PairBatch:
    """같은 cut을 공유하는 sibling pair batch.

    a/b는 같은 original의 서로 다른 variant이며 variant_id가 반드시 다릅니다.
    has_sibling[i]가 False이면 b가 a의 자리표시자이므로 within loss에서 제외합니다.
    """

    cut: int
    input_a: PredictorInput
    input_b: PredictorInput
    target_a: Tensor  # [B, P, 2, H] = V_a[cut]
    target_b: Tensor  # [B, P, 2, H] = V_b[cut]
    future_state_diff_a: Tensor  # [B, L+1, P, H] = D_a, rollout 채점 전용
    future_state_diff_b: Tensor
    var_state_a: Tensor  # [B, L+1, P, H] 전체 variant state, rollout 채점 전용
    var_state_b: Tensor
    valid: Tensor  # [B, P] bool
    has_sibling: Tensor  # [B] bool
    is_identity_a: Tensor  # [B] bool
    original_ids: list[str]
    variant_ids_a: list[str]
    variant_ids_b: list[str]


# --------------------------------------------------------------------------------------
# Synthetic Page 생성
# --------------------------------------------------------------------------------------


def _shared_dynamics(cfg: SyntheticConfig, generator: torch.Generator) -> dict[str, Tensor]:
    """모든 original이 공유하는 update rule.

    stream별 mixing matrix 하나를 모든 depth가 공유하고, depth별로는 scale만 다릅니다.
    이렇게 하면 update 흐름이 배울 수 있는 구조를 가지면서도 depth별 전개와 stream 구분은
    그대로 남습니다. 마지막 FFN stream은 zero-scale로 두어 inactive 처리를 검증합니다.
    """
    hidden = cfg.hidden_size
    mix = torch.randn(hidden, hidden, generator=generator) / math.sqrt(hidden)
    ffn = torch.randn(hidden, hidden, generator=generator) / math.sqrt(hidden)
    scale = torch.linspace(0.25, 0.45, cfg.n_blocks).view(-1, 1).repeat(1, 2)
    scale[cfg.n_blocks - 1, 1] = 0.0  # zero-scale 대조
    return {"mix": mix, "ffn": ffn, "scale": scale}


def _run_dynamics(
    h0: Tensor, dyn: dict[str, Tensor], noise: Tensor | None
) -> tuple[Tensor, Tensor]:
    """residual identity를 정확히 만족하는 state/updates를 굴립니다.

    h0: [P, H]. 반환 state [L+1, P, H], updates [L, P, 2, H].
    """
    n_blocks = int(dyn["scale"].shape[0])
    states = [h0]
    updates = []
    for d in range(n_blocks):
        h = states[-1]
        u_mix = torch.tanh(h @ dyn["mix"]) * dyn["scale"][d, 0]
        u_ffn = torch.tanh(h @ dyn["ffn"]) * dyn["scale"][d, 1]
        if noise is not None:
            u_mix = u_mix + noise[d, :, 0]
            u_ffn = u_ffn + noise[d, :, 1] * float(dyn["scale"][d, 1] > 0)
        stacked = torch.stack([u_mix, u_ffn], dim=1)  # [P, 2, H]
        updates.append(stacked)
        states.append(h + stacked.sum(dim=1))  # exact residual identity
    return torch.stack(states), torch.stack(updates)


def _landmark_positions(n_landmarks: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """증가하는 token offset과 relative position. 마지막은 canonical final prompt token."""
    gaps = torch.randint(1, 6, (n_landmarks,), generator=generator)
    offsets = torch.cumsum(gaps, dim=0)
    prompt_len = int(offsets[-1]) + 3
    offsets[-1] = prompt_len - 1  # canonical final prompt token
    rel = offsets.float() / max(prompt_len - 1, 1)
    return offsets.long(), rel.float()


def _corrupt_invalid(
    state: Tensor, updates: Tensor, valid: Tensor, generator: torch.Generator
) -> None:
    """invalid landmark 자리를 residual identity를 깨는 junk로 채웁니다.

    mask 처리를 건너뛰는 코드가 있으면 결과가 달라지므로 바로 드러납니다.
    """
    bad = ~valid
    if not bool(bad.any()):
        return
    state[:, bad] = torch.randn(state[:, bad].shape, generator=generator) * 3.0
    updates[:, bad] = torch.randn(updates[:, bad].shape, generator=generator) * 3.0


def make_synthetic_split(
    cfg: SyntheticConfig,
    split: str,
    n_originals: int,
    seed: int,
    delta_scale: float = 5.0,
    perturbation_family: int = 0,
    dyn: dict[str, Tensor] | None = None,
) -> tuple[PageDataset, dict[str, Tensor]]:
    """한 split의 synthetic original group들을 만듭니다.

    perturbation_family는 variant delta를 뽑는 subspace를 고릅니다. unseen 판정은
    train에서 쓰지 않은 family를 쓰는 것으로 표현합니다.
    """
    generator = torch.Generator().manual_seed(seed)
    if dyn is None:
        dyn = _shared_dynamics(cfg, torch.Generator().manual_seed(12345))
    hidden, n_p = cfg.hidden_size, cfg.n_landmarks
    # family별 고정 delta basis. family 0/1은 train/known, 2는 unseen perturbation.
    basis_gen = torch.Generator().manual_seed(9000 + perturbation_family)
    basis = torch.randn(8, hidden, generator=basis_gen)
    basis = basis / basis.norm(dim=-1, keepdim=True)

    groups: list[OriginalGroup] = []
    provenance = {
        "source": "synthetic",
        "split": split,
        "perturbation_family": perturbation_family,
        "delta_scale": delta_scale,
        "hidden_size": hidden,
        "n_blocks": cfg.n_blocks,
        "n_landmarks": n_p,
        "model_hash": "synthetic-none",
        "tokenizer_hash": "synthetic-none",
        "config_hash": "synthetic-none",
    }
    for i in range(n_originals):
        original_id = f"{split}-orig-{i:04d}"
        h0 = torch.randn(n_p, hidden, generator=generator) * 0.8
        offsets, rel = _landmark_positions(n_p, generator)
        valid = torch.rand(n_p, generator=generator) > cfg.invalid_landmark_prob
        valid[0] = True
        valid[-1] = True  # canonical final prompt token은 항상 유효합니다.
        state, updates = _run_dynamics(h0, dyn, noise=None)
        _corrupt_invalid(state, updates, valid, generator)
        original = Page(
            state=state,
            updates=updates,
            valid=valid,
            token_offsets=offsets,
            relative_positions=rel,
            original_id=original_id,
            variant_id=f"{original_id}#orig",
            is_identity=False,
            provenance=dict(provenance, role="original"),
        )
        variants: list[Page] = []
        n_variants = cfg.variants_per_original
        # identity variant를 결정적으로 배치해 비율이 흔들리지 않게 합니다.
        # 전체 example 중 identity 비율이 identity_fraction이 되도록 주기를 잡습니다.
        period = max(
            int(round(1.0 / max(cfg.identity_fraction * cfg.variants_per_original, 1e-6))), 1
        )
        n_identity = 1 if (cfg.identity_fraction > 0 and i % period == 0) else 0
        for v in range(n_variants):
            is_identity = v == 0 and n_identity == 1
            if is_identity:
                v_state, v_updates = state.clone(), updates.clone()
                v_valid = valid.clone()
            else:
                coeff = torch.randn(basis.shape[0], generator=generator)
                delta = (coeff @ basis) * (delta_scale / math.sqrt(basis.shape[0]))
                # landmark별로 다른 세기로 주입해 위치별 전개가 달라지게 합니다.
                weight = torch.rand(n_p, 1, generator=generator) * 0.8 + 0.2
                h0_var = h0 + delta.view(1, -1) * weight
                noise = torch.randn(
                    cfg.n_blocks, n_p, 2, hidden, generator=generator
                ) * cfg.noise_scale
                v_state, v_updates = _run_dynamics(h0_var, dyn, noise=noise)
                v_valid = valid.clone()
                if bool(v_valid.sum() > 2) and torch.rand(1, generator=generator).item() < 0.2:
                    drop = int(torch.nonzero(v_valid)[1])
                    v_valid[drop] = False
                _corrupt_invalid(v_state, v_updates, v_valid, generator)
            variants.append(
                Page(
                    state=v_state,
                    updates=v_updates,
                    valid=v_valid,
                    token_offsets=offsets.clone(),
                    relative_positions=rel.clone(),
                    original_id=original_id,
                    variant_id=f"{original_id}#var{v:02d}",
                    is_identity=is_identity,
                    provenance=dict(provenance, role="variant", identity=is_identity),
                )
            )
        groups.append(OriginalGroup(original_id, original, variants))
    return PageDataset(split=split, groups=groups), dyn


def make_synthetic_dataset(cfg: Config) -> dict[str, PageDataset]:
    """E0용 5개 split을 만듭니다. 같은 original의 variants는 같은 split에 둡니다."""
    syn = cfg.data.synthetic
    dyn = _shared_dynamics(syn, torch.Generator().manual_seed(12345))
    base = cfg.run.seed
    delta, harder = syn.delta_scale, syn.harder_delta_scale
    plan = [
        ("train", syn.n_originals_train, base + 1, delta, 0),
        ("validation", syn.n_originals_validation, base + 2, delta, 0),
        ("known_test", syn.n_originals_test, base + 3, delta, 0),
        ("unseen_perturbation_test", syn.n_originals_test, base + 4, delta, 2),
        ("harder_test", syn.n_originals_test, base + 5, harder, 2),
    ]
    out = {}
    for split, n, seed, delta, family in plan:
        dataset, _ = make_synthetic_split(
            syn, split, n, seed, delta_scale=delta, perturbation_family=family, dyn=dyn
        )
        out[split] = dataset
    return out


def save_dataset(datasets: dict[str, PageDataset], directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index = {}
    for split, dataset in datasets.items():
        pages: list[Page] = []
        layout = []
        for group in dataset.groups:
            layout.append(
                {
                    "original_id": group.original_id,
                    "n_variants": len(group.variants),
                }
            )
            pages.append(group.original)
            pages.extend(group.variants)
        save_pages(pages, directory / f"{split}.npz")
        index[split] = {"layout": layout, "data_hash": dataset.data_hash()}
    (directory / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return directory


def load_dataset(directory: str | Path) -> dict[str, PageDataset]:
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    datasets = {}
    for split, entry in index.items():
        pages = load_pages(directory / f"{split}.npz")
        groups = []
        cursor = 0
        for item in entry["layout"]:
            original = pages[cursor]
            n_var = item["n_variants"]
            variants = pages[cursor + 1 : cursor + 1 + n_var]
            cursor += 1 + n_var
            groups.append(OriginalGroup(item["original_id"], original, variants))
        dataset = PageDataset(split=split, groups=groups)
        if dataset.data_hash() != entry["data_hash"]:
            raise ValueError(f"data hash mismatch for split {split}")
        datasets[split] = dataset
    return datasets


# --------------------------------------------------------------------------------------
# Normalization: train originals만 사용
# --------------------------------------------------------------------------------------


@dataclass
class NormStats:
    """train originals만으로 계산해 freeze한 scale 모음.

    역할별로 따로 보관합니다.
      input_state_scale  [L+1]  : predictor 입력 state
      input_update_scale [L, 2] : predictor 입력 update
      target_scale       [L, 2] : V target (variant - original update)
      sibling_scale      [L, 2] : V_a - V_b
      rollout_scale      [L+1]  : D = state_variant - state_original
    작은 분모로 신호를 과장하지 않도록 floor 아래의 scale은 inactive로 표시합니다.
    """

    input_state_scale: Tensor
    input_update_scale: Tensor
    target_scale: Tensor
    sibling_scale: Tensor
    rollout_scale: Tensor
    floor: float
    source: str = "train_originals"
    n_originals: int = 0

    @property
    def target_active(self) -> Tensor:
        return self.target_scale > self.floor

    @property
    def rollout_active(self) -> Tensor:
        return self.rollout_scale > self.floor

    def _safe(self, scale: Tensor) -> Tensor:
        return scale.clamp_min(self.floor)

    def norm_state(self, state: Tensor) -> Tensor:
        """[..., D, P, H] / input_state_scale[:D]."""
        depth = state.shape[-3]
        return state / self._safe(self.input_state_scale[:depth]).view(-1, 1, 1)

    def norm_update(self, updates: Tensor) -> Tensor:
        """[..., D, P, 2, H] / input_update_scale[:D]."""
        depth = updates.shape[-4]
        return updates / self._safe(self.input_update_scale[:depth]).view(-1, 1, 2, 1)

    def norm_target(self, target: Tensor, cut: int) -> Tensor:
        """[..., P, 2, H] / target_scale[cut]."""
        return target / self._safe(self.target_scale[cut]).view(1, 2, 1)

    def denorm_target(self, target: Tensor, cut: int) -> Tensor:
        return target * self._safe(self.target_scale[cut]).view(1, 2, 1)

    def norm_sibling(self, diff: Tensor, cut: int) -> Tensor:
        return diff / self._safe(self.sibling_scale[cut]).view(1, 2, 1)

    def norm_rollout(self, diff: Tensor, depth: int) -> Tensor:
        """[..., P, H] / rollout_scale[depth]."""
        return diff / self._safe(self.rollout_scale[depth])

    def state_dict(self) -> dict:
        return {
            "input_state_scale": self.input_state_scale,
            "input_update_scale": self.input_update_scale,
            "target_scale": self.target_scale,
            "sibling_scale": self.sibling_scale,
            "rollout_scale": self.rollout_scale,
            "floor": self.floor,
            "source": self.source,
            "n_originals": self.n_originals,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> NormStats:
        return cls(
            input_state_scale=payload["input_state_scale"],
            input_update_scale=payload["input_update_scale"],
            target_scale=payload["target_scale"],
            sibling_scale=payload["sibling_scale"],
            rollout_scale=payload["rollout_scale"],
            floor=float(payload["floor"]),
            source=payload.get("source", "train_originals"),
            n_originals=int(payload.get("n_originals", 0)),
        )

    def hash(self) -> str:
        blob = b"".join(
            t.numpy().tobytes()
            for t in (
                self.input_state_scale,
                self.input_update_scale,
                self.target_scale,
                self.sibling_scale,
                self.rollout_scale,
            )
        )
        return hashlib.sha256(blob).hexdigest()[:16]


def compute_norm_stats(train: PageDataset, floor: float = 1e-6) -> NormStats:
    """train originals의 valid landmark에서 depth별 RMS를 구합니다.

    변형 정보를 전혀 쓰지 않습니다. target/sibling scale도 original update의 RMS를
    그대로 씁니다: V는 update와 같은 단위이므로 typical update 크기 대비 상대오차가
    되고, variant 통계를 미리 들여다보지 않게 됩니다.
    """
    n_blocks = train.n_blocks
    state_sq = torch.zeros(n_blocks + 1)
    state_n = torch.zeros(n_blocks + 1)
    upd_sq = torch.zeros(n_blocks, 2)
    upd_n = torch.zeros(n_blocks, 2)
    for group in train.groups:
        page = group.original
        mask = page.valid
        st = page.state[:, mask]  # [L+1, Pv, H]
        state_sq += st.pow(2).sum(dim=(1, 2))
        state_n += st.shape[1] * st.shape[2]
        up = page.updates[:, mask]  # [L, Pv, 2, H]
        upd_sq += up.pow(2).sum(dim=(1, 3))
        upd_n += up.shape[1] * up.shape[3]
    state_scale = (state_sq / state_n.clamp_min(1)).sqrt()
    update_scale = (upd_sq / upd_n.clamp_min(1)).sqrt()
    # floor 미만은 zero/noise scale로 보고 inactive 처리합니다.
    state_scale = torch.where(state_scale > floor, state_scale, torch.zeros_like(state_scale))
    update_scale = torch.where(update_scale > floor, update_scale, torch.zeros_like(update_scale))
    return NormStats(
        input_state_scale=state_scale,
        input_update_scale=update_scale,
        target_scale=update_scale.clone(),
        sibling_scale=update_scale.clone(),
        rollout_scale=state_scale.clone(),
        floor=floor,
        n_originals=len(train.groups),
    )


# --------------------------------------------------------------------------------------
# Sibling sampling과 batching
# --------------------------------------------------------------------------------------


@dataclass
class PairSample:
    group: OriginalGroup
    index_a: int
    index_b: int | None
    cut: int


def sample_pairs(
    dataset: PageDataset,
    generator: torch.Generator,
    cuts_per_pair: int,
    group_indices: list[int] | None = None,
) -> list[PairSample]:
    """original 하나당 sibling pair를 뽑습니다.

    a,b는 variant_id가 서로 다른 두 variant입니다. identity variant를 두 개의
    서로 다른 sibling으로 세지 않습니다 (같은 index를 a,b로 쓰지 않습니다).
    """
    indices = group_indices if group_indices is not None else list(range(len(dataset.groups)))
    n_blocks = dataset.n_blocks
    samples: list[PairSample] = []
    for gi in indices:
        group = dataset.groups[gi]
        n_var = len(group.variants)
        for _ in range(cuts_per_pair):
            cut = int(torch.randint(0, n_blocks, (1,), generator=generator))
            if n_var >= 2:
                perm = torch.randperm(n_var, generator=generator)
                ia, ib = int(perm[0]), int(perm[1])
            else:
                ia, ib = 0, None
            samples.append(PairSample(group=group, index_a=ia, index_b=ib, cut=cut))
    return samples


def enumerate_pairs(dataset: PageDataset, cuts: list[int]) -> list[PairSample]:
    """결정적 평가용: 모든 group의 모든 sibling 조합과 지정 cut을 나열합니다."""
    samples = []
    for group in dataset.groups:
        n_var = len(group.variants)
        for ia in range(n_var):
            # sibling은 바로 다음 variant로 고정합니다 (n_var == 1이면 sibling 없음).
            ib = (ia + 1) % n_var if n_var >= 2 else None
            for cut in cuts:
                samples.append(PairSample(group, ia, ib, cut))
    return samples


def collate(samples: list[PairSample]) -> PairBatch:
    """같은 cut을 가진 sample들을 하나의 PairBatch로 묶습니다."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    cut = samples[0].cut
    if any(s.cut != cut for s in samples):
        raise ValueError("collate requires a single shared cut")

    orig_state, orig_updates = [], []
    var_state_a, var_state_b = [], []
    var_upd_a, var_upd_b = [], []
    valid_list, rel_list, off_list = [], [], []
    has_sibling, identity_a = [], []
    original_ids, ids_a, ids_b = [], [], []

    for sample in samples:
        group = sample.group
        page_a = group.variants[sample.index_a]
        page_b = group.variants[sample.index_b] if sample.index_b is not None else page_a
        if sample.index_b is not None and page_a.variant_id == page_b.variant_id:
            raise ValueError("sibling pair must use two distinct variant ids")
        orig_state.append(group.original.state)
        orig_updates.append(group.original.updates)
        var_state_a.append(page_a.state)
        var_state_b.append(page_b.state)
        var_upd_a.append(page_a.updates)
        var_upd_b.append(page_b.updates)
        common = group.original.valid & page_a.valid & page_b.valid
        valid_list.append(common)
        rel_list.append(group.original.relative_positions)
        off_list.append(group.original.token_offsets)
        has_sibling.append(sample.index_b is not None)
        identity_a.append(page_a.is_identity)
        original_ids.append(group.original_id)
        ids_a.append(page_a.variant_id)
        ids_b.append(page_b.variant_id)

    orig_state_t = torch.stack(orig_state)
    orig_updates_t = torch.stack(orig_updates)
    var_state_a_t = torch.stack(var_state_a)
    var_state_b_t = torch.stack(var_state_b)
    var_upd_a_t = torch.stack(var_upd_a)
    var_upd_b_t = torch.stack(var_upd_b)
    valid_t = torch.stack(valid_list)
    rel_t = torch.stack(rel_list)
    off_t = torch.stack(off_list)

    def build_input(v_state: Tensor, v_upd: Tensor) -> PredictorInput:
        # prefix만 잘라 담습니다: state 0..cut, updates 0..cut-1.
        return PredictorInput(
            cut=cut,
            orig_state=orig_state_t,
            orig_updates=orig_updates_t,
            var_state_prefix=v_state[:, : cut + 1].clone(),
            var_updates_prefix=v_upd[:, :cut].clone(),
            valid=valid_t,
            relative_positions=rel_t,
            token_offsets=off_t,
        )

    return PairBatch(
        cut=cut,
        input_a=build_input(var_state_a_t, var_upd_a_t),
        input_b=build_input(var_state_b_t, var_upd_b_t),
        target_a=(var_upd_a_t - orig_updates_t)[:, cut].clone(),
        target_b=(var_upd_b_t - orig_updates_t)[:, cut].clone(),
        future_state_diff_a=(var_state_a_t - orig_state_t).clone(),
        future_state_diff_b=(var_state_b_t - orig_state_t).clone(),
        var_state_a=var_state_a_t,
        var_state_b=var_state_b_t,
        valid=valid_t,
        has_sibling=torch.tensor(has_sibling),
        is_identity_a=torch.tensor(identity_a),
        original_ids=original_ids,
        variant_ids_a=ids_a,
        variant_ids_b=ids_b,
    )


def group_by_cut(samples: list[PairSample]) -> list[list[PairSample]]:
    """cut별로 묶어 각 forward가 하나의 cut만 다루게 합니다."""
    buckets: dict[int, list[PairSample]] = {}
    for sample in samples:
        buckets.setdefault(sample.cut, []).append(sample)
    return [buckets[cut] for cut in sorted(buckets)]


def swap_support(batch: PairBatch, generator: torch.Generator) -> PairBatch:
    """support-swap 대조군: 같은 original의 sibling 사이에서 support를 뒤섞습니다.

    a의 관측 prefix를 b의 prefix로 바꾸고 target은 a의 것을 유지합니다. original은
    그대로이므로 group 밖으로 나가지 않습니다.
    """
    swapped = PredictorInput(
        cut=batch.cut,
        orig_state=batch.input_a.orig_state,
        orig_updates=batch.input_a.orig_updates,
        var_state_prefix=batch.input_b.var_state_prefix,
        var_updates_prefix=batch.input_b.var_updates_prefix,
        valid=batch.input_a.valid,
        relative_positions=batch.input_a.relative_positions,
        token_offsets=batch.input_a.token_offsets,
    )
    return PairBatch(
        cut=batch.cut,
        input_a=swapped,
        input_b=batch.input_b,
        target_a=batch.target_a,
        target_b=batch.target_b,
        future_state_diff_a=batch.future_state_diff_a,
        future_state_diff_b=batch.future_state_diff_b,
        var_state_a=batch.var_state_a,
        var_state_b=batch.var_state_b,
        valid=batch.valid,
        has_sibling=batch.has_sibling,
        is_identity_a=batch.is_identity_a,
        original_ids=batch.original_ids,
        variant_ids_a=batch.variant_ids_a,
        variant_ids_b=batch.variant_ids_b,
    )
