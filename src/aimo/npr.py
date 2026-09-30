"""Native next-token response(NPR) loss.

teacher는 target model의 실제 next-token distribution `p_t = softmax(W_lm N(h) + b_lm)`입니다.
primary loss는 **coarse categorical KL**입니다.

    categories = teacher top-k token (k=32) + OTHER
    p_t(OTHER) = 1 - Σ_{i in top-k} p_t(i)   (나머지 vocabulary의 실제 확률 질량)
    L = Σ_{c in categories} p_t(c) (log p_t(c) - log p_s(c))

top-k를 다시 합 1로 정규화하지 않고, 이 값을 full KL이라고 부르지 않습니다. teacher와 student의
full normalizer(logsumexp)는 vocabulary chunk로 계산하며 T × V tensor를 상시 보관하지
않습니다. student 쪽은 chunk마다 gradient checkpoint를 써서 backward 때도 T × V activation을
저장하지 않습니다. `full_vocab_kl`은 audit 경로입니다.

token 평균은 문제마다 먼저 내고 그다음 문제끼리 평균합니다. 긴 문제의 token 수를 독립
표본 수처럼 취급하지 않습니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

COARSE_KL_NAME = "coarse_topk_other_kl"
FULL_KL_NAME = "full_vocab_kl_audit"
DEFAULT_TOPK = 32
LOG_FLOOR = math.log(1e-30)


def _linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    out = x @ weight.T
    return out if bias is None else out + bias


def log1mexp(x: Tensor) -> Tensor:
    """log(1 - exp(x)) for x <= 0, 수치적으로 안정한 형태."""
    x = x.clamp(max=-1e-12)
    return torch.where(x > -math.log(2.0), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


@torch.no_grad()
def teacher_topk_other(
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    k: int = DEFAULT_TOPK,
    chunk: int = 8192,
) -> TeacherTopK:
    """teacher top-k log-prob와 OTHER log-mass를 vocabulary chunk로 계산합니다.

    hidden [N, H'] (final norm 이후 hidden), weight [V, H'], bias [V] 또는 None.
    running logsumexp와 running top-k만 유지합니다.
    """
    n = hidden.shape[0]
    vocab = weight.shape[0]
    if k >= vocab:
        raise ValueError(f"top-k={k} must be smaller than the vocabulary ({vocab})")
    top_val = torch.full((n, k), float("-inf"), device=hidden.device)
    top_idx = torch.zeros((n, k), dtype=torch.long, device=hidden.device)
    lse = torch.full((n,), float("-inf"), device=hidden.device)
    for start in range(0, vocab, chunk):
        end = min(start + chunk, vocab)
        logits = _linear(hidden.float(), weight[start:end].float(),
                         None if bias is None else bias[start:end].float())
        lse = torch.logaddexp(lse, torch.logsumexp(logits, dim=-1))
        cand_val = torch.cat([top_val, logits], dim=-1)
        cand_idx = torch.cat(
            [top_idx, torch.arange(start, end, device=hidden.device).expand(n, -1)], dim=-1)
        top_val, pos = cand_val.topk(k, dim=-1)
        top_idx = cand_idx.gather(-1, pos)
    logp = top_val - lse[:, None]
    log_other = log1mexp(torch.logsumexp(logp, dim=-1)).clamp_min(LOG_FLOOR)
    return TeacherTopK(indices=top_idx, logp=logp, log_other=log_other, logsumexp=lse)


@dataclass
class TeacherTopK:
    indices: Tensor  # [N, k]
    logp: Tensor  # [N, k] full-softmax 기준 log-prob (재정규화하지 않음)
    log_other: Tensor  # [N] 나머지 vocabulary의 실제 log-mass
    logsumexp: Tensor  # [N]

    def other_mass(self) -> Tensor:
        return self.log_other.exp()

    def to(self, device) -> TeacherTopK:
        return TeacherTopK(*(t.to(device) for t in (self.indices, self.logp, self.log_other,
                                                    self.logsumexp)))

    def select(self, rows: Tensor) -> TeacherTopK:
        return TeacherTopK(self.indices[rows], self.logp[rows], self.log_other[rows],
                           self.logsumexp[rows])


def _chunk_lse(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return torch.logsumexp(x @ weight.T + bias, dim=-1)


def chunked_logsumexp(
    x: Tensor, weight: Tensor, bias: Tensor, chunk: int = 8192, *, save_memory: bool = True
) -> Tensor:
    """전체 vocabulary logsumexp. gradient가 필요하면 chunk마다 checkpoint로 재계산합니다."""
    parts = []
    for start in range(0, weight.shape[0], chunk):
        w, b = weight[start : start + chunk], bias[start : start + chunk]
        if save_memory and torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
            parts.append(checkpoint(_chunk_lse, x, w, b, use_reentrant=False))
        else:
            parts.append(_chunk_lse(x, w, b))
    return torch.logsumexp(torch.stack(parts, dim=-1), dim=-1)


def student_coarse(
    z: Tensor, decoder: Tensor, bias: Tensor, teacher: TeacherTopK, chunk: int = 8192
) -> tuple[Tensor, Tensor]:
    """student의 teacher-top-k log-prob [N, k]와 OTHER log-mass [N].

    student logit = D z + b. top-k 위치는 decoder row를 gather해 직접 계산합니다.
    """
    lse = chunked_logsumexp(z, decoder, bias, chunk)
    rows = decoder[teacher.indices]  # [N, k, r]
    logits_k = torch.einsum("nkr,nr->nk", rows, z) + bias[teacher.indices]
    logp_k = logits_k - lse[:, None]
    log_other = log1mexp(torch.logsumexp(logp_k, dim=-1)).clamp_min(LOG_FLOOR)
    return logp_k, log_other


def coarse_kl(teacher: TeacherTopK, student_logp_k: Tensor, student_log_other: Tensor) -> Tensor:
    """token별 coarse categorical KL [N] (top-k + OTHER, 재정규화 없음)."""
    p_k = teacher.logp.exp()
    p_other = teacher.log_other.exp()
    term_k = (p_k * (teacher.logp - student_logp_k)).sum(-1)
    term_other = p_other * (teacher.log_other - student_log_other)
    return term_k + term_other


@torch.no_grad()
def full_vocab_kl(
    t_hidden: Tensor,
    t_weight: Tensor,
    t_bias: Tensor | None,
    s_hidden: Tensor,
    s_weight: Tensor,
    s_bias: Tensor | None,
    chunk: int = 8192,
) -> Tensor:
    """audit용 full-vocabulary KL(p_t || p_s) [N]. normalizer 계산 후 chunk로 누적합니다."""
    vocab = t_weight.shape[0]
    zero_t = torch.zeros(vocab, device=t_hidden.device) if t_bias is None else t_bias.float()
    zero_s = torch.zeros(vocab, device=s_hidden.device) if s_bias is None else s_bias.float()
    lse_t = chunked_logsumexp(t_hidden.float(), t_weight.float(), zero_t, chunk, save_memory=False)
    lse_s = chunked_logsumexp(s_hidden.float(), s_weight.float(), zero_s, chunk, save_memory=False)
    total = torch.zeros(t_hidden.shape[0], device=t_hidden.device)
    for start in range(0, vocab, chunk):
        end = min(start + chunk, vocab)
        lt = _linear(t_hidden.float(), t_weight[start:end].float(), zero_t[start:end])
        ls = _linear(s_hidden.float(), s_weight[start:end].float(), zero_s[start:end])
        lt, ls = lt - lse_t[:, None], ls - lse_s[:, None]
        total += (lt.exp() * (lt - ls)).sum(-1)
    return total


def problem_normalized_mean(values: Tensor, mask: Tensor) -> Tensor:
    """values [B, ..., T], mask [B, T] -> 문제별 valid-token 평균의 문제 평균.

    stage 축 같은 중간 축은 문제 안에서 함께 평균합니다.
    """
    mask_f = mask.to(values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(1)
    mask_f = mask_f.expand_as(values)
    dims = tuple(range(1, values.ndim))
    per_problem = (values * mask_f).sum(dims) / mask_f.sum(dims).clamp_min(1.0)
    has_tokens = mask.flatten(1).any(dim=1)
    if not bool(has_tokens.any()):
        raise ValueError("no valid tokens in the batch")
    return per_problem[has_tokens].mean()
