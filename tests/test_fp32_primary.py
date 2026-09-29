import json
from types import SimpleNamespace
import pytest
from aimo.fp32_runtime import Guard,policy
from aimo.fp32_collect import collect
from aimo.bf16_collect import identity,slot_path

def write_policy(root,backend='fp32_sdpa'):
    (root/'protocol').mkdir()
    p={'profile':{'numerical_backend':backend,'scorer_version':'2'},
       'precision':{'tf32_matmul':False,'tf32_cudnn':False},'cohort_id':root.name,
       'primary_dtype':'float32','behavior_policy_id':'fp32-only',
       'observation_provenance':{'tokenizer_hash':'tok','template_hash':'tpl'}}
    (root/'protocol/policy.json').write_text(json.dumps(p));return p

def test_no_elapsed_time_cutoff(tmp_path):
    (tmp_path/'gpu_budget.json').write_text(json.dumps({'active_seconds':100*3600}))
    assert not Guard(tmp_path,120).should_stop()[0]
    (tmp_path/'STOP').write_text('stop')
    assert Guard(tmp_path).should_stop()[0]

def test_bf16_resume_rejected(tmp_path):
    write_policy(tmp_path,'bf16_sdpa')
    with pytest.raises(AssertionError):policy(tmp_path)

def test_interrupted_slot_never_resampled(tmp_path):
    p=write_policy(tmp_path);(tmp_path/'collection').mkdir()
    pr={'prompt_id':'p','prompt':'q','gold':'4','input_ids':[1],'input_token_hash':'h',
        'slots':[{'slot_id':'s','seed':9}]}
    ident=identity(pr,pr['slots'][0],p)
    path=slot_path(tmp_path,'collection','s');path.parent.mkdir()
    path.write_text(json.dumps({'identity':ident,'state':'started','original_id':'o','prompt_id':'p'}))
    import torch
    backend=SimpleNamespace(render=lambda _: {'input_ids':torch.tensor([[1]])},
                            generate_batch=lambda *a,**k:pytest.fail('must not resample'))
    collect(tmp_path,'collection',{'original_id':'o'},pr,backend)
    result=json.loads(path.read_text())
    assert result['state']=='interrupted' and result['outcome']=='U_score'
    assert result['termination']=='interrupted'


def test_fp32_training_wrapper_real_tiny_run(tmp_path,monkeypatch,datasets):
    import aimo.fp32_post as fp
    import importlib
    tm=importlib.import_module('aimo.train')
    from aimo.config import config_from_dict
    from helpers import tiny_payload
    cfg=config_from_dict(tiny_payload(model={'name':'joint'},train={'task':'joint'}))
    cfg.train.lr=3e-4;cfg.train.weight_decay=1e-3;cfg.train.flow_weight=.1
    monkeypatch.setattr(fp,'datasets',lambda root:datasets)
    monkeypatch.setattr(fp,'config',lambda root:cfg)
    orig=tm.train
    def short(config,*a,**kw):
        config.train.max_epochs=2;config.train.patience=2
        return orig(config,*a,**kw)
    monkeypatch.setattr(tm,'train',short)
    fp.train_worker(tmp_path,'joint',1)
    report=json.loads((tmp_path/'training/joint_seed1/precision.json').read_text())
    assert report['observed']['gradients']==['torch.float32']
    assert report['observed']['inputs']==['torch.float32']


def test_batch1_marks_only_dispatched_slot_started(tmp_path,monkeypatch):
    import aimo.fp32_collect as mod
    import torch
    write_policy(tmp_path);(tmp_path/'collection').mkdir()
    monkeypatch.setattr(mod,'memory_settings',lambda r:{'batch_size':1,'execution_id':'test'})
    for name in ['reset_peak_memory_stats','empty_cache']:
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ['memory_allocated','max_memory_allocated','max_memory_reserved']:
        monkeypatch.setattr(torch.cuda,name,lambda:100)
    pr={'prompt_id':'p','prompt':'q','gold':'4','input_ids':[1],'input_token_hash':'h',
        'slots':[{'slot_id':'s0','seed':1},{'slot_id':'s1','seed':2}]}
    def fail(requests,**kwargs):
        assert len(requests)==1 and requests[0].slot_id=='s0'
        assert slot_path(tmp_path,'collection','s0').exists()
        assert not slot_path(tmp_path,'collection','s1').exists()
        raise RuntimeError('simulated worker crash')
    backend=SimpleNamespace(render=lambda _: {'input_ids':torch.tensor([[1]])},generate_batch=fail)
    with pytest.raises(RuntimeError,match='simulated'):
        mod.collect(tmp_path,'collection',{'original_id':'o'},pr,backend)
    assert json.loads(slot_path(tmp_path,'collection','s0').read_text())['state']=='started'
    assert not slot_path(tmp_path,'collection','s1').exists()

def test_same_policy_accepts_smaller_runtime_batch(tmp_path):
    import aimo.fp32_runtime as rt
    p=write_policy(tmp_path);(tmp_path/'runtime').mkdir()
    (tmp_path/'runtime/memory_settings.json').write_text(json.dumps({
        'batch_size':1,'replicas':4,'policy_hash':p['behavior_policy_id'],
        'dtype':'float32','max_new_tokens':16384,'max_total_context':40960}))
    assert rt.memory_settings(tmp_path)['batch_size']==1
    assert rt.policy(tmp_path)['behavior_policy_id']=='fp32-only'


def _memory_smoke_fixture(root,monkeypatch):
    import aimo.fp32_recovery as recovery
    from aimo.real_run import sha
    a=root/'runtime/oom_recovery_001';a.mkdir(parents=True)
    (root/'runtime/memory_batches').mkdir()
    ids=[f'smoke-{i}' for i in range(12)]
    (a/'smoke_plan.json').write_text(json.dumps({'slot_ids':ids}))
    for sid in ids:
        p=slot_path(root,'collection',sid);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'state':'completed','termination':'token_cap' if sid==ids[-1] else 'eos',
            'identity':{'policy_hash':'p'},'outcome':'X' if sid==ids[-1] else 'C'}))
        (root/'runtime/memory_batches'/(sha(sid)+'.json')).write_text(json.dumps({
            'batch_size':1,'kv_dtypes':['torch.float32'],'logits_dtype':'torch.float32',
            'before_allocated_bytes':1000,'after_allocated_bytes':1000,
            'peak_vram_bytes':20*2**30,'peak_reserved_bytes':22*2**30,
            'max_cache_bytes':4*2**30,'generated_tokens':100,'wall_seconds':10}))
    monkeypatch.setattr(recovery,'memory_settings',lambda r:{'policy_hash':'p'})
    monkeypatch.setattr(recovery,'preserve_check',lambda r:True)
    return recovery,ids

def test_memory_gate_accepts_bounded_batches_and_natural_cap(tmp_path,monkeypatch):
    recovery,ids=_memory_smoke_fixture(tmp_path,monkeypatch)
    result=recovery.check_smoke(tmp_path)
    assert result['passed'] and result['consecutive_batches']==12
    assert result['outcomes']=={'C':11,'X':1}

@pytest.mark.parametrize('field,value',[
    ('after_allocated_bytes',128*2**20),('peak_vram_bytes',34*2**30)])
def test_memory_gate_blocks_leak_or_unsafe_peak(tmp_path,monkeypatch,field,value):
    from aimo.real_run import sha
    recovery,ids=_memory_smoke_fixture(tmp_path,monkeypatch)
    p=tmp_path/'runtime/memory_batches'/(sha(ids[-1])+'.json')
    data=json.loads(p.read_text());data[field]=value;p.write_text(json.dumps(data))
    with pytest.raises(AssertionError):recovery.check_smoke(tmp_path)
    assert not (tmp_path/'runtime/oom_recovery_001/smoke_passed.json').exists()
