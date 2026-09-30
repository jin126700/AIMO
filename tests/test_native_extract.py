"""native all-token macro extractor와 NativePage schema test (CPU toy model)."""

from __future__ import annotations

import json

import pytest
import torch

from aimo.macro_page import macro_boundaries
from aimo.native_extract import (
    AuditSink,
    PromptPolicy,
    extract_native_pages,
    model_provenance,
    pad_right,
    render_prompt,
    run_extraction,
    tokenize_prompt,
)
from aimo.native_page import (
    NATIVE_PAGE_SCHEMA,
    SEG_PROBE,
    SEG_PROBLEM,
    SEG_TEMPLATE,
    DataLimit,
    SchemaIncompatible,
    load_native_pages,
    pad_native,
    require_native,
    save_native_pages,
)
from aimo.native_toy import ToyTokenizer, build_toy_hybrid
from aimo.page import Page

TEXTS = [
    "If 3x + 4 = 10,  find x.",
    "Let  n be\nan integer with n > 2. Compute n^2 - 1 for n = 5.",
    "Find $a+b$ if a = 2.",
]


def _records(texts=TEXTS):
    return [dict(problem_id=f"p{i}", root_id=f"r{i}", split="train", text=t)
            for i, t in enumerate(texts)]


def _extract(model, n_macro=4, batch_size=1, texts=TEXTS):
    tok = ToyTokenizer()
    policy = PromptPolicy(max_tokens=512)
    prov = model_provenance(model, tok, n_macro, policy, model_id="toy", model_revision="rev",
                            tokenizer_revision="rev")
    return extract_native_pages(model, tok, _records(texts), prov, policy, batch_size=batch_size)


def test_qwen35_style_32_layers_group_by_four():
    assert macro_boundaries(32, 8) == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    # 다른 model은 실제 layer 수를 읽습니다. 모든 model을 32층으로 강제하지 않습니다.
    assert macro_boundaries(28, 8) == [0, 3, 7, 10, 14, 17, 21, 24, 28]
    assert macro_boundaries(36, 8)[-1] == 36


def test_macro_sums_residual_identity_and_all_tokens():
    model = build_toy_hybrid(n_layers=8)
    pages, summary = _extract(model)
    assert summary["max_identity_error"] < 1e-5
    # module hook 출력도 actual residual update와 일치해야 합니다 (non-tautological check).
    assert summary["max_module_disagreement_mixer"] < 1e-4
    assert summary["max_module_disagreement_ffn"] < 1e-4
    tok = ToyTokenizer()
    for page, text in zip(pages, TEXTS, strict=True):
        rendered = render_prompt(tok, text, PromptPolicy())
        assert page.seq_len == len(tok(rendered.text)["input_ids"])  # 모든 token 유지
        assert page.state.shape == (5, page.seq_len, 16)
        assert page.mixer.shape == page.ffn.shape == (4, page.seq_len, 16)
    assert pages[0].provenance.macro_boundaries == [0, 2, 4, 6, 8]
    kinds = pages[0].provenance.mixer_kinds
    assert "linear_attention" in kinds and "full_attention" in kinds


def test_hybrid_mixer_both_kinds_sum_exactly_per_macro():
    """macro update는 layer별 actual update의 합입니다 (G=L로 layer별 값을 따로 뽑아 비교)."""
    model = build_toy_hybrid(n_layers=8)
    fine, _ = _extract(model, n_macro=8)
    macro, _ = _extract(model, n_macro=4)
    for f, m in zip(fine, macro, strict=True):
        assert torch.allclose(m.mixer, f.mixer.view(4, 2, *f.mixer.shape[1:]).sum(1), atol=1e-5)
        assert torch.allclose(m.ffn, f.ffn.view(4, 2, *f.ffn.shape[1:]).sum(1), atol=1e-5)


def test_update_outside_module_is_detected_not_hidden():
    """Mixer 출력에 module 밖에서 scale이 붙으면 module hook은 틀리고 residual 차이는 맞습니다."""
    model = build_toy_hybrid(n_layers=8, outside_scale=0.5)
    _, summary = _extract(model)
    assert summary["max_identity_error"] < 1e-5
    assert summary["max_module_disagreement_mixer"] > 0.5


def test_post_norm_layout_uses_normed_sublayer_outputs():
    model = build_toy_hybrid(n_layers=8, layout="post_norm")
    _, summary = _extract(model)
    assert summary["max_identity_error"] < 1e-5
    assert summary["max_module_disagreement_mixer"] < 1e-4


def test_final_norm_is_separate_from_raw_final_boundary():
    model = build_toy_hybrid(n_layers=8)
    pages, summary = _extract(model)
    assert summary["final_norm_distinct_from_raw_boundary"]
    tok = ToyTokenizer()
    enc = tokenize_prompt(tok, render_prompt(tok, TEXTS[0], PromptPolicy()), 512)
    with torch.no_grad():
        normed = model.model(enc["input_ids"][None]).last_hidden_state[0]
    assert torch.allclose(pages[0].final_norm, normed, atol=1e-5)
    assert not torch.allclose(pages[0].state[-1], normed, atol=1e-3)


def test_variable_length_padding_matches_single_sequence():
    model = build_toy_hybrid(n_layers=8)
    single, _ = _extract(model, batch_size=1)
    batched, _ = _extract(model, batch_size=3)
    assert len({p.seq_len for p in single}) > 1
    for a, b in zip(single, batched, strict=True):
        assert a.seq_len == b.seq_len
        assert torch.allclose(a.state, b.state, atol=1e-5)
        assert torch.allclose(a.mixer, b.mixer, atol=1e-5)
    batch = pad_native(single)
    assert batch.mask.sum(1).tolist() == [p.seq_len for p in single]


def test_no_silent_truncation():
    tok = ToyTokenizer()
    rendered = render_prompt(tok, "x " * 400, PromptPolicy())
    with pytest.raises(DataLimit):
        tokenize_prompt(tok, rendered, max_tokens=64)


def test_span_map_marks_problem_template_and_probe_tokens():
    tok = ToyTokenizer()
    text = "If 3x + 4 = 10, find x."
    rendered = render_prompt(tok, text, PromptPolicy(), probe_text=" so x = 2")
    enc = tokenize_prompt(tok, rendered, 512)
    seg, spans = enc["segments"], enc["char_spans"]
    problem_tokens = (seg == SEG_PROBLEM).nonzero().flatten()
    rebuilt = "".join(text[int(spans[i, 0]) : int(spans[i, 1])] for i in problem_tokens)
    assert rebuilt == text
    assert int((seg == SEG_TEMPLATE).sum()) > 0 and int((seg == SEG_PROBE).sum()) > 0


def test_unknown_layout_is_schema_incompatible():
    model = build_toy_hybrid(n_layers=4)
    model.config.model_type = "some_unreviewed_arch"
    ids, mask = pad_right([torch.tensor([3, 4, 5])], 0)
    with pytest.raises(SchemaIncompatible):
        run_extraction(model, ids, mask, AuditSink(2), 2)


def test_legacy_sparse_page_is_not_converted(tmp_path):
    legacy = Page(
        state=torch.zeros(3, 17, 8), updates=torch.zeros(2, 17, 2, 8),
        valid=torch.ones(17, dtype=torch.bool), token_offsets=torch.arange(17),
        relative_positions=torch.zeros(17), original_id="o", variant_id="v",
    )
    with pytest.raises(SchemaIncompatible, match="SCHEMA_INCOMPATIBLE"):
        require_native(legacy)
    (tmp_path / "index.json").write_text(json.dumps({"schema": "aimo-page-store-v2"}))
    with pytest.raises(SchemaIncompatible):
        load_native_pages(tmp_path)


def test_native_store_roundtrip_and_tamper_detection(tmp_path):
    pages, _ = _extract(build_toy_hybrid(n_layers=8))
    save_native_pages(pages, tmp_path)
    index = json.loads((tmp_path / "native_index.json").read_text())
    assert index["schema"] == NATIVE_PAGE_SCHEMA
    meta = index["pages"][0]["provenance"]
    for key in ("model_id", "model_revision", "tokenizer_revision", "n_layers", "hidden_size",
                "macro_boundaries", "observation", "dtype", "backend", "chat_template_hash",
                "prompt_policy", "dataset"):
        assert key in meta
    loaded = load_native_pages(tmp_path)
    assert torch.equal(loaded[1].state, pages[1].state)
    blob = torch.load(tmp_path / "native_pages.pt", weights_only=True)
    blob["0/state"] = blob["0/state"] + 1
    torch.save(blob, tmp_path / "native_pages.pt")
    with pytest.raises(SchemaIncompatible, match="sha256"):
        load_native_pages(tmp_path)


def test_page_rejects_projected_hidden_size():
    pages, _ = _extract(build_toy_hybrid(n_layers=8))
    page = pages[0]
    page.provenance.hidden_size = 17
    with pytest.raises(SchemaIncompatible, match="native H"):
        page.validate()


def test_streaming_sink_matches_offline_encoder_without_raw_pages():
    """제출 경로의 저차원 누적이 offline audit + Stage-E encoder와 같은 값을 냅니다."""
    from aimo.native_extract import StreamSink
    from aimo.sketch import fold_head, make_sketch
    from aimo.stage_e import StageE

    model = build_toy_hybrid(n_layers=8)
    pages, _ = _extract(model)
    stage = StageE(4, 16, 4, 48, seed=1)
    with torch.no_grad():
        stage.lower_raw.normal_(0, 0.2)
        stage.offset.normal_()
    folded = fold_head(model, make_sketch(48, 6, 2), 2)
    tok = ToyTokenizer()
    enc = tokenize_prompt(tok, render_prompt(tok, TEXTS[1], PromptPolicy()), 512)
    ids = enc["input_ids"][None]
    with torch.no_grad():
        sink = StreamSink(stage.encoder_matrix(), stage.offset, folded)
        sink.last_index = torch.tensor([ids.shape[1] - 1])
        run_extraction(model, ids, torch.ones_like(ids), sink, 4)
        stream = sink.stacked()
        offline = stage.encode_page(pages[1].state, pages[1].mixer, pages[1].ffn)
    for key in ("z_state", "z_in", "z_mixer", "z_ffn"):
        assert torch.allclose(stream[key][0], offline[key], atol=1e-4), key
    assert torch.allclose(stream["z_state"] - stream["z_in"],
                          stream["z_mixer"] + stream["z_ffn"], atol=1e-4)
    assert torch.allclose(sink.y[0], folded(pages[1].final_norm), atol=1e-4)
    assert torch.allclose(sink.final_norm_last[0], pages[1].final_norm[-1], atol=1e-5)
