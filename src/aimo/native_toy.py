"""CPU 검증 전용 toy decoder와 toy tokenizer.

로컬에는 Qwen3.5 weights와 그것을 읽는 transformers 버전이 없습니다. 그래서 hybrid Mixer
구조(linear-attention Mixer와 full-attention Mixer가 섞인 pre-norm layer, 그리고 OLMo 식
post-norm layer)를 같은 module 이름으로 흉내 낸 작은 random-init model로 hook 논리만
검증합니다. 이 model의 결과는 실제 model 검증이 아닙니다.
"""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import torch
from torch import Tensor, nn


class ToyConfig:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ToyLinearMixer(nn.Module):
    """causal linear attention + output gate. 출력 tensor 하나를 돌려줍니다."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        q = torch.nn.functional.elu(self.q(x)) + 1
        k = torch.nn.functional.elu(self.k(x)) + 1
        v = self.v(x)
        if attention_mask is not None:
            keep = attention_mask.unsqueeze(-1).to(x.dtype)
            k, v = k * keep, v * keep
        kv = torch.cumsum(k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)  # [B,T,H,H]
        norm = torch.cumsum(k, dim=1)
        denom = (q * norm).sum(-1, keepdim=True).clamp_min(1e-6)
        out = torch.einsum("bth,bthd->btd", q, kv) / denom
        return self.o(out * torch.sigmoid(self.gate(x)))


class ToyFullAttention(nn.Module):
    """causal softmax attention + output gate. HF처럼 (output, weights) tuple을 돌려줍니다."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.gate = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        t = x.shape[1]
        scores = q @ k.transpose(1, 2) / q.shape[-1] ** 0.5
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        allowed = causal.unsqueeze(0)
        if attention_mask is not None:
            allowed = allowed & attention_mask.bool().unsqueeze(1)
        scores = scores.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        return self.o((attn @ v) * torch.sigmoid(self.gate(x))), None


class ToyMLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, 2 * hidden, bias=False)
        self.gate = nn.Linear(hidden, 2 * hidden, bias=False)
        self.down = nn.Linear(2 * hidden, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class ToyPreNormLayer(nn.Module):
    """Qwen3.5 식 pre-norm layer. mixer는 `linear_attn` 또는 `self_attn` 중 하나입니다.

    `outside_scale`이 있으면 Mixer 출력에 module 밖에서 scale을 곱해 residual에 더합니다.
    module hook 출력이 actual update와 달라지는 경우를 검출하는 test 전용 설정입니다.
    """

    def __init__(self, hidden: int, mixer: str, outside_scale: float | None = None) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden)
        if mixer == "linear_attention":
            self.linear_attn = ToyLinearMixer(hidden)
        else:
            self.self_attn = ToyFullAttention(hidden)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.mlp = ToyMLP(hidden)
        self.outside_scale = outside_scale

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None, **_) -> tuple:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        if hasattr(self, "linear_attn"):
            update = self.linear_attn(normed, attention_mask)
        else:
            update, _ = self.self_attn(normed, attention_mask)
        if self.outside_scale is not None:
            update = update * self.outside_scale
        hidden_states = residual + update
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return (hidden_states,)


class ToyPostNormLayer(nn.Module):
    """OLMo-2 식 post-norm layer. update는 sublayer 출력에 norm을 적용한 값입니다."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = ToyFullAttention(hidden)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.mlp = ToyMLP(hidden)
        self.post_feedforward_layernorm = RMSNorm(hidden)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None, **_) -> tuple:
        residual = hidden_states
        attn, _ = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + self.post_attention_layernorm(attn)
        residual = hidden_states
        hidden_states = residual + self.post_feedforward_layernorm(self.mlp(hidden_states))
        return (hidden_states,)


class ToyBackbone(nn.Module):
    def __init__(self, config: ToyConfig, layers: list[nn.Module]) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
        **_,
    ) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask)[0]
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class ToyHybridForCausalLM(nn.Module):
    def __init__(self, config: ToyConfig, layers: list[nn.Module], head_bias: bool) -> None:
        super().__init__()
        self.config = config
        self.model = ToyBackbone(config, layers)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=head_bias)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, **_
    ) -> SimpleNamespace:
        hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


def build_toy_hybrid(
    hidden_size: int = 16,
    n_layers: int = 8,
    vocab_size: int = 48,
    layout: str = "pre_norm",
    seed: int = 0,
    head_bias: bool = True,
    outside_scale: float | None = None,
) -> ToyHybridForCausalLM:
    """linear-attention과 full-attention Mixer를 3:1로 섞은 toy decoder."""
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        layer_types = [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n_layers)
        ]
        if layout == "pre_norm":
            layers = [ToyPreNormLayer(hidden_size, kind, outside_scale) for kind in layer_types]
        elif layout == "post_norm":
            layer_types = ["full_attention"] * n_layers
            layers = [ToyPostNormLayer(hidden_size) for _ in range(n_layers)]
        else:
            raise ValueError(f"unknown toy layout {layout!r}")
        config = ToyConfig(
            model_type=f"aimo_toy_{layout}",
            num_hidden_layers=n_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            layer_types=layer_types,
            final_logit_softcapping=None,
        )
        model = ToyHybridForCausalLM(config, layers, head_bias)
        # random init에서 residual stream이 너무 작지 않도록 embedding scale을 키웁니다.
        with torch.no_grad():
            model.model.embed_tokens.weight.mul_(2.0)
    finally:
        torch.random.set_rng_state(generator_state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


_TOKEN_RE = re.compile(r"\s+|\d|[A-Za-z]+|[^\sA-Za-z\d]")


class ToyTokenizer:
    """offset mapping을 주는 toy tokenizer. whitespace run은 별도 token입니다."""

    SPECIAL = ("<|pad|>", "<|user|>", "<|assistant|>")
    chat_template = "<|user|>\n{problem}\n<|assistant|>\n"
    name_or_path = "aimo/toy-tokenizer"
    token_pattern = _TOKEN_RE

    def __init__(self, vocab_size: int = 48) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def _token_id(self, piece: str) -> int:
        if piece in self.SPECIAL:
            return self.SPECIAL.index(piece)
        digest = hashlib.sha256(piece.encode()).digest()
        return len(self.SPECIAL) + int.from_bytes(digest[:4], "big") % (
            self.vocab_size - len(self.SPECIAL)
        )

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = True, **_
    ) -> str:
        problem = messages[-1]["content"]
        return self.chat_template.format(problem=problem)

    def __call__(self, text: str, return_offsets_mapping: bool = False, add_special_tokens=False):
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        specials = re.compile("|".join(re.escape(s) for s in self.SPECIAL))
        while cursor < len(text):
            match = specials.match(text, cursor) or self.token_pattern.match(text, cursor)
            piece = match.group(0)
            ids.append(self._token_id(piece))
            offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out
