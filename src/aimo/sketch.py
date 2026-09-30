"""제출용 logit-contrast sketch와 native head folding.

고정 sketch `R ∈ R^[V × q]` (q=64, 고정 seed)는 column 합이 0입니다 (`R^T 1 = 0`).
raw native logits `l = W_lm N(h_final) + b_lm`에 대해

    y = R^T log_softmax(l) = R^T l = (R^T W_lm) N(h_final) + R^T b_lm

이므로 native head를 `[q, H]`로 folding해 두면 vocabulary 전체 logits 없이 y를 얻습니다.
student는 `y_hat = (R^T D) z_state + R^T b`입니다.

이 값은 full NPR가 아니라 **logit-contrast sketch**입니다. 압축 사각지대를 보려고 독립 sketch
`R'`, full-vocabulary KL(`npr.full_vocab_kl`), 짧은 수학 continuation audit를 따로 둡니다.
logit softcap처럼 head 뒤에 비선형 처리가 있으면 folding identity가 성립하지 않으므로 거부합니다.
평가 문제를 보고 sketch나 normalization을 다시 맞추지 않습니다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

SKETCH_NAME = "logit_contrast_sketch"
DEFAULT_SKETCH_Q = 64
DEFAULT_SKETCH_SEED = 20260930
DEFAULT_AUDIT_SEED = 20260931

# head 뒤의 비선형 처리. 이 config 값이 켜져 있으면 folding을 거부합니다.
NONLINEAR_HEAD_ATTRS = ("final_logit_softcapping", "logit_softcapping", "attn_logit_softcapping")


class FoldingInvalid(ValueError):
    """native head가 linear가 아니어서 folding identity를 쓸 수 없습니다."""


def make_sketch(vocab_size: int, q: int = DEFAULT_SKETCH_Q, seed: int = DEFAULT_SKETCH_SEED,
                chunk: int = 65536) -> Tensor:
    """R [V, q] float32. float64로 만들고 column 평균을 빼서 R^T 1 = 0을 맞춥니다."""
    generator = torch.Generator().manual_seed(seed)
    parts = [torch.randn(min(chunk, vocab_size - start), q, generator=generator,
                         dtype=torch.float64)
             for start in range(0, vocab_size, chunk)]
    sketch = torch.cat(parts) / q**0.5
    sketch -= sketch.mean(dim=0, keepdim=True)
    return sketch.float()


def sketch_hash(sketch: Tensor) -> str:
    return hashlib.sha256(sketch.contiguous().numpy().tobytes()).hexdigest()[:16]


def column_sum_error(sketch: Tensor) -> float:
    """max |R^T 1| / max |R|. float32 저장 오차 수준이어야 합니다."""
    return float(sketch.double().sum(0).abs().max() / sketch.abs().max())


def check_head_linear(config) -> None:
    config = getattr(config, "text_config", None) or config
    for attr in NONLINEAR_HEAD_ATTRS[:2]:
        if getattr(config, attr, None):
            raise FoldingInvalid(
                f"{attr}={getattr(config, attr)} makes the head nonlinear; the folding identity "
                "R^T log_softmax(l) = (R^T W) N(h) + R^T b does not hold"
            )


@dataclass
class FoldedHead:
    """native head의 sketch folding. cached target model에서 한 번 만들고 재사용합니다."""

    weight: Tensor  # [q, H] = R^T W_lm
    bias: Tensor  # [q] = R^T b_lm (bias가 없으면 0)
    q: int
    seed: int
    source_dtype: str

    def __call__(self, normed_hidden: Tensor) -> Tensor:
        w = self.weight.to(normed_hidden.device)
        return normed_hidden.float() @ w.T + self.bias.to(normed_hidden.device)


def output_head(model: nn.Module) -> nn.Linear:
    head = getattr(model, "lm_head", None) or model.get_output_embeddings()
    if head is None:
        raise FoldingInvalid("model has no output head")
    return head


@torch.no_grad()
def fold_head(model: nn.Module, sketch: Tensor, seed: int, chunk: int = 32768) -> FoldedHead:
    """R^T W_lm, R^T b_lm를 vocabulary chunk로 float32 누적합니다."""
    check_head_linear(getattr(model, "config", None))
    head = output_head(model)
    weight = head.weight
    vocab, hidden = weight.shape
    if sketch.shape[0] != vocab:
        raise FoldingInvalid(f"sketch rows {sketch.shape[0]} != vocab {vocab}")
    q = sketch.shape[1]
    folded = torch.zeros(q, hidden, device=weight.device)
    bias = torch.zeros(q, device=weight.device)
    for start in range(0, vocab, chunk):
        r = sketch[start : start + chunk].to(weight.device)
        folded += r.T @ weight[start : start + chunk].float()
        if head.bias is not None:
            bias += r.T @ head.bias[start : start + chunk].float()
    return FoldedHead(folded.cpu(), bias.cpu(), q, seed, str(weight.dtype).replace("torch.", ""))


@torch.no_grad()
def verify_folding(
    logits_fn: Callable[[], tuple[Tensor, Tensor]], folded: FoldedHead, sketch: Tensor
) -> dict:
    """R^T log_softmax(l)와 folded head 출력이 같은지 한 입력에서 확인합니다.

    logits_fn() -> (full logits [T, V], final-normed hidden [T, H]).
    """
    logits, normed = logits_fn()
    reference = torch.log_softmax(logits.float(), dim=-1) @ sketch.to(logits.device)
    estimate = folded(normed)
    err = float((reference - estimate).norm() / reference.norm().clamp_min(1e-12))
    return {"relative_error": err, "sketch": SKETCH_NAME, "q": folded.q}


def fold_student(decoder: Tensor, bias: Tensor, sketch: Tensor) -> tuple[Tensor, Tensor]:
    """(R^T D [q, r], R^T b [q])."""
    return sketch.T @ decoder, sketch.T @ bias
