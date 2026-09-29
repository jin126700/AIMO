"""실제 source allowlist / metric / frozen gate / resume 회귀."""
import copy
import json
from pathlib import Path

import pytest
import torch

from aimo.config import LRTConfig
from aimo.lrt import build_pair_tensors, fit_macro_norm_stats
from aimo.lrt_experiment import (
    Rank4LinearTransport, make_toy_rank4, swap_donor_index, train_lrt,
    transport_loss, transport_term,
)
from aimo.lrt_real import (
    bridge_decision, decision, evaluate_fine, freeze_floor, load_fp32_page_source,
    macro_pairs, source_manifest, unique_originals,
)
from aimo.macro_page import relation_path_energy
from aimo.page import save_pages
from aimo.flow_representation import digest, split_ids


def fixture_source(tmp_path):
    pairs = make_toy_rank4(n_originals=4, n_variants=1)['pairs']
    source = tmp_path/'source'
    (source/'pages'/'artifacts').mkdir(parents=True)
    (source/'collection').mkdir(); (source/'protocol').mkdir()
    prov = {'forward_dtype': 'float32', 'tf32': False, 'autocast': False,
            'model_hash': 'model', 'policy_hash': 'policy'}
    groups, index = [], {}
    for i, (o, v) in enumerate(pairs):
        o.variant_id = o.original_id
        prompts = []
        for j, p in enumerate([o,v]):
            p.provenance = dict(prov)
            path = source/'pages'/'artifacts'/f'{i}-{j}.npz'
            save_pages([p], path)
            index[p.variant_id] = {'path': str(path), 'sha256': digest(path),
                                   'original_id': o.original_id, 'split': 'train'}
            prompts.append({'prompt_id': p.variant_id})
        groups.append({'original_id': o.original_id, 'split': 'train', 'prompts': prompts})
    for name, obj in [('collection/groups.json', groups), ('pages/index.json', index),
                      ('protocol/policy.json', {'observation_provenance': prov})]:
        (source/name).write_text(json.dumps(obj))
    return source


def test_source_allowlist_checksum_provenance_immutable(tmp_path, monkeypatch):
    source = fixture_source(tmp_path)
    before = {str(p):digest(p) for p in source.rglob('*') if p.is_file()}
    real_open = Path.open; seen = []
    def guarded(self, *args, **kwargs):
        seen.append(str(self))
        assert not any(x in str(self) for x in ['labels/', 'outcomes/'])
        return real_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    m = source_manifest(source); ids = list(m['originals'])
    loaded = load_fp32_page_source(m, ids[:1])
    assert len(loaded) == 1
    unopened = {r['path'] for oid in ids[1:] for r in m['originals'][oid]['pages']}
    assert not (set(seen) & unopened)
    assert before == {str(p):digest(p) for p in source.rglob('*') if p.is_file()}
    bad = copy.deepcopy(m); bad['originals'][ids[0]]['pages'][0]['sha256'] = 'bad'
    with pytest.raises(ValueError, match='checksum'): load_fp32_page_source(bad, ids[:1])
    bad = copy.deepcopy(m); bad['expected_provenance']['model_hash'] = 'other'
    with pytest.raises(ValueError, match='provenance'): load_fp32_page_source(bad, ids[:1])


def test_scalar_metric_landmark_count_and_gradient():
    x = torch.ones(2, 3, 2, 5)*2; valid = torch.tensor([True, False, True])
    pred = torch.zeros_like(x, requires_grad=True)
    term = transport_term(x, pred, valid, .1)
    assert term.mse_zero == 4 and term.ratio == 1
    assert term.n_scalar == 40 and term.n_cells == 8
    assert transport_term(x[:, :1], pred[:, :1], valid[:1], .1).mse_zero == 4
    loss = transport_loss(x, pred, valid, .1); loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[:, 1].eq(0).all()


def test_floor_train_only_and_freeze(tmp_path):
    pairs = macro_pairs(make_toy_rank4(n_originals=4, n_variants=1)['pairs'])
    train = pairs[:2]; ids = [o.original_id for o,v in train]
    stats = fit_macro_norm_stats(unique_originals(train))
    tau = freeze_floor(tmp_path, train, stats, ids)
    manifest = json.loads((tmp_path/'floor_manifest.json').read_text())
    assert tau == max(1e-12, manifest['q05']) and manifest['number_positive'] == 8
    for o,v in pairs[2:]: v.updates.mul_(1000)
    assert freeze_floor(tmp_path, train, stats, ids) == tau
    with pytest.raises(ValueError, match='frozen'): freeze_floor(tmp_path, pairs, stats, ids)


def test_relation_path_cancellation():
    o,v = make_toy_rank4(n_originals=2, n_variants=1)['pairs'][0]
    v.updates = o.updates.clone(); v.updates[0] += 1; v.updates[1] -= 1
    pe = relation_path_energy(o,v,[0,2,32])
    assert torch.all(pe[0] > 0)
    assert (v.updates-o.updates)[:2].sum(0).abs().max() < 2e-7


def test_fine_bridge_regions_and_original_split():
    pairs = macro_pairs(make_toy_rank4(n_originals=4, n_variants=1)['pairs'], fine=True)
    stats = fit_macro_norm_stats(unique_originals(pairs[:2]))
    rank = Rank4LinearTransport.fit(pairs[:2],stats)
    result = evaluate_fine(pairs[2:],stats,rank,1e-6)
    assert len(result['records']) == 8
    assert [r['query_layers'] for r in result['records'][:4]] == [list(range(i,i+8)) for i in range(0,32,8)]
    train,dev = split_ids([str(i) for i in range(32)])
    assert len(train)==28 and len(dev)==4 and not set(train)&set(dev)
    assert bridge_decision(.97,1.) == 'MACROPAGE_COMPRESSION_LIMIT'
    assert bridge_decision(.99,1.1) == 'PROCEED_MACRO_LRT'


def test_gate_requires_two_positive_swaps_and_cross_original():
    rows = [{'transport_ratio': .9, 'swap_gap': gap} for gap in [.1,-.01,-.01]]
    assert not decision(rows, 1.)['validation_gate_B']
    rows[1]['swap_gap'] = .1
    assert decision(rows, 1.)['decision'] == 'LRT_ADDS_OVER_LINEAR'
    with pytest.raises(ValueError, match='cross-original'): swap_donor_index(0,['a','a'],0)


def test_epoch_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(1)
    pairs = macro_pairs(make_toy_rank4(n_originals=4,n_variants=1,n_landmarks=2,hidden=8)['pairs'])
    stats = fit_macro_norm_stats(unique_originals(pairs[:2])); cfg=LRTConfig()
    full=train_lrt(pairs[:2],pairs[2:],stats,cfg,floor=1e-6,max_epochs=2,seed=0)
    train_lrt(pairs[:2],pairs[2:],stats,cfg,floor=1e-6,max_epochs=1,seed=0,checkpoint_dir=tmp_path)
    resumed=train_lrt(pairs[:2],pairs[2:],stats,cfg,floor=1e-6,max_epochs=2,seed=0,checkpoint_dir=tmp_path)
    assert full['history'] == resumed['history']
    assert all(torch.equal(v,resumed['model'].state_dict()[k]) for k,v in full['model'].state_dict().items())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_requested_cuda_device():
    pairs=macro_pairs(make_toy_rank4(n_originals=2,n_variants=1)['pairs'])
    stats=fit_macro_norm_stats(unique_originals(pairs)).to(torch.device('cuda'))
    payload=build_pair_tensors(*pairs[0],stats)
    assert all(v.device.type=='cuda' for v in payload.values())
    assert pairs[0][0].updates.device.type=='cpu'


def test_real_prepare_freezes_three_seeds_and_no_external_pages(tmp_path, monkeypatch):
    from aimo.lrt_real import prepare
    source=fixture_source(tmp_path)
    groups=json.loads((source/'collection/groups.json').read_text())
    # metadata만 추가; 외부 Page 파일을 열면 존재하지 않아 실패합니다.
    index=json.loads((source/'pages/index.json').read_text())
    for name in ['validation','known_original_test']:
        oid=f'held-{name}'
        groups.append({'original_id':oid,'split':name,'prompts':[{'prompt_id':oid}]})
        index[oid]={'original_id':oid,'split':name,'path':str(source/'pages'/'artifacts'/f'{oid}.npz'),'sha256':'unopened'}
    (source/'collection/groups.json').write_text(json.dumps(groups))
    (source/'pages/index.json').write_text(json.dumps(index))
    root=tmp_path/'run'
    prepare(source,root)
    spec=json.loads((root/'spec.json').read_text())
    assert spec['train_seeds']==[0,1,2] and spec['eval_seed']==0
    assert spec['bridge_split']=='lrt_dev'
    before=digest(root/'floor_manifest.json')
    prepare(source,root)
    assert before==digest(root/'floor_manifest.json')
