"""E-FLOW-1 label isolation / pooling / frozen encoder contracts."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from helpers import tiny_payload
from aimo.config import config_from_dict
from aimo.data import make_synthetic_dataset,compute_norm_stats
from aimo.model import build_model
from aimo.flow_representation import split_ids,stage1_train,assert_label_free,digest
from aimo.flow_probe import panel_pool,criterion,shuffle_labels,fit_probe,probe_suite

def setup():
    cfg=config_from_dict(tiny_payload(train={'task':'flow','select_metric':'flow_total','max_epochs':1}))
    ds=make_synthetic_dataset(cfg)
    for d in ds.values():
        for g in d.groups:g.pair_labels={};g.panel_label=None
    return cfg,ds

def test_split_original_and_test_exclusion():
    ids=[str(i) for i in range(32)];a,b=split_ids(ids)
    assert len(a)==28 and len(b)==4 and not set(a)&set(b)
    assert split_ids(ids[::-1])==(a,b)
    assert not (set(a)|set(b))&{'test-original'}

def test_stage1_does_not_read_behavior_or_select_by_labels(tmp_path,monkeypatch):
    import aimo.train as tm
    cfg,ds=setup();cfg.paths.output_root=str(tmp_path)
    selected={k:ds[k] for k in ['train','validation']}
    def forbidden(*a,**kw):raise AssertionError('behavior path accessed')
    monkeypatch.setattr(tm,'_behavior_terms',forbidden)
    original=Path.open
    def guard(p,*a,**kw):
        assert '/labels/' not in str(p),'behavior label file opened'
        return original(p,*a,**kw)
    monkeypatch.setattr(Path,'open',guard)
    result=stage1_train(cfg,selected)
    assert result['behavior_supervision_seen'] is False
    assert result['supervision']['select_metric']=='flow_total'
    from aimo.train import load_checkpoint
    _,stats,_=load_checkpoint(cfg.run_dir/'best.pt')
    assert stats.hash()==compute_norm_stats(ds['train']).hash()

def test_stage1_rejects_labels():
    cfg,ds=setup();ds['train'].groups[0].pair_labels={'bad':object()}
    with pytest.raises(ValueError):assert_label_free(ds)

def test_pair_determinism_hash_and_frozen_probe():
    cfg,ds=setup();d=ds['train'];g=d.groups[0];stats=compute_norm_stats(d)
    torch.manual_seed(0);m=build_model(cfg,d.hidden_size,d.n_blocks,d.n_landmarks).eval().requires_grad_(False)
    a=m.encode_pair(g.original,g.variants[0],stats)
    b=m.encode_pair(g.original,g.variants[0],stats)
    assert torch.equal(a,b)
    import hashlib
    assert hashlib.sha256(a.numpy().tobytes()).digest()==hashlib.sha256(b.numpy().tobytes()).digest()
    before={k:v.clone() for k,v in m.state_dict().items()}
    z=np.tile(a.numpy(),(8,1));z[:,0]=np.arange(8)
    fit_probe(z,np.array([0]*4+[1]*4),z,np.array([0]*4+[1]*4),z,[.1])
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in m.parameters())
    # Original-only path cannot read variant information through masked reference cells.
    c=m.encode_pair(g.original,g.variants[1],stats,original_only=True)
    e=m.encode_pair(g.original,g.original,stats,original_only=True)
    assert torch.equal(c,e)

def test_panel_permutation_singleton_and_missing():
    z=np.random.default_rng(0).normal(size=(4,128)).astype('float32')
    np.testing.assert_array_equal(panel_pool(z),panel_pool(z[::-1]))
    assert np.all(panel_pool(z[:1])[128:]==0)
    with pytest.raises(ValueError):panel_pool(np.empty((0,128)))

def test_criterion_bounds_and_unlabeled_exclusion():
    policy={'source':'synthetic test only','definition_id':'test','robust_threshold':.1,'nonrobust_threshold':.3}
    assert criterion([.15,.25],True,policy)==(None,'ambiguous')
    assert criterion([0,.5],False,policy)==(None,'unresolved')
    assert criterion([0,.05],False,policy)==(1,'robust')
    assert criterion([.4,.5],False,policy)==(0,'nonrobust')
    assert criterion([0,0],True,None)==(None,'unresolved')
    result,_=probe_suite([],[],{'panels':[],'criterion':None},{})
    assert result['status']=='ROBUSTNESS_PROBE_DATA_LIMIT' and result['shuffle_executed']==0

def test_random_control_not_load_checkpoint(monkeypatch):
    cfg,ds=setup();d=ds['train']
    def forbidden(*a,**k):raise AssertionError('checkpoint loaded')
    monkeypatch.setattr(torch,'load',forbidden)
    torch.manual_seed(0)
    m=build_model(cfg,d.hidden_size,d.n_blocks,d.n_landmarks)
    assert m.core.d_model==16

def test_shuffle_by_original_only():
    y={f'o{i}':i%2 for i in range(20)}
    s=shuffle_labels(y,7)
    assert set(s)==set(y) and sorted(s.values())==sorted(y.values()) and s!=y
    assert shuffle_labels(y,7)==s

def test_cpu_synthetic_pretraining_probe_sanity(tmp_path):
    """합성 dynamics sanity: 실제 robustness evidence와 분리된 고정 fixture."""
    from aimo.train import load_checkpoint
    from aimo.flow_representation import extract
    from sklearn.metrics import brier_score_loss
    torch.set_num_threads(2)
    cfg=config_from_dict(tiny_payload(
        paths={'output_root':str(tmp_path)},run={'run_id':'sanity','device':'cpu'},
        data={'synthetic':{'n_originals_train':32,'n_originals_validation':16,'n_originals_test':32,
             'variants_per_original':2,'invalid_landmark_prob':0.,'identity_fraction':0.,
             'unresolved_fraction':0.,'noise_scale':0.}},
        train={'task':'flow','select_metric':'flow_total','max_epochs':40,'patience':10,
               'batch_originals':8,'microbatch_originals':8,'cuts_per_pair':2}))
    ds=make_synthetic_dataset(cfg)
    ys={s:np.array([g.panel_label.robust_label for g in d.groups]) for s,d in ds.items()}
    for d in ds.values():
        for g in d.groups:g.pair_labels={};g.panel_label=None;g.metadata={}
    stage1_train(cfg,{s:ds[s] for s in ['train','validation']})
    m,stats,_=load_checkpoint(cfg.run_dir/'best.pt')
    _,panels=extract(m,stats,{s:ds[s] for s in ['train','validation','known_test']})
    torch.manual_seed(0);d=ds['train']
    random=build_model(cfg,d.hidden_size,d.n_blocks,d.n_landmarks)
    _,rp=extract(random,stats,{s:ds[s] for s in ['train','validation','known_test']})
    scores={}
    for name,rows in [('flow',panels),('random',rp)]:
        x={s:np.asarray([p['z'] for p in rows if p['split']==s]) for s in ['train','validation','known_test']}
        pred,_=fit_probe(x['train'],ys['train'],x['validation'],ys['validation'],x['known_test'],[.01,.1,1.])
        scores[name]=brier_score_loss(ys['known_test'],pred)
    print('synthetic fixed fixture Brier',scores)
    assert scores['flow']<scores['random'],scores

def test_representation_reload_hash_stable(tmp_path):
    import hashlib
    cfg,ds=setup();d=ds['train'];g=d.groups[0];stats=compute_norm_stats(d)
    torch.manual_seed(4);model=build_model(cfg,d.hidden_size,d.n_blocks,d.n_landmarks).eval()
    z=model.encode_pair(g.original,g.variants[0],stats).numpy()
    path=tmp_path/'model.pt';torch.save(model.state_dict(),path)
    other=build_model(cfg,d.hidden_size,d.n_blocks,d.n_landmarks).eval()
    other.load_state_dict(torch.load(path,weights_only=True))
    z2=other.encode_pair(g.original,g.variants[0],stats).numpy()
    assert hashlib.sha256(z.tobytes()).hexdigest()==hashlib.sha256(z2.tobytes()).hexdigest()

def test_incomplete_panel_cannot_be_falsely_robust(tmp_path):
    from aimo.flow_probe import labels_from_source
    (tmp_path/'labels').mkdir()
    (tmp_path/'labels/collection_pairs.json').write_text(json.dumps([
        {'original_id':'o','signed_drop':0.,'drop_lower':0.,'drop_upper':0.,'semantic_valid':'verified'}]))
    rows=[{'original_id':'o','split':'train','coverage':{'expected':2,'actual':2}}]
    policy={'source':'test only','definition_id':'test','robust_threshold':.1,'nonrobust_threshold':.3}
    result=labels_from_source(tmp_path,rows,policy)
    assert result['panels'][0]['label'] is None
    assert result['panels'][0]['max_drop_bounds']==[0.,1.]

def test_linear_probe_controls_and_paired_bootstrap():
    rows=[];labels=[]
    for split,n in [('train',12),('validation',6),('known_test',6)]:
        for i in range(n):
            y=i%2;ident=split+str(i);z=[float(y),float(i)/n]
            rows.append({'original_id':ident,'split':split,'z':z,'m0':z,'raw':z})
            labels.append({'original_id':ident,'split':split,'label':y})
    spec={'probe_C':[.1],'shuffle_repeats':20}
    result,boot=probe_suite(rows,rows,{'panels':labels,'criterion':{'source':'synthetic'}},spec)
    assert result['shuffle_executed']==20 and len(boot)==3
    assert result['models']['Flow']['metrics']['balanced_accuracy']==1.
    assert boot['Random']['brier_improvement_ci95']==[0.,0.]
