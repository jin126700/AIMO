"""FP32 label 연결, 사전 고정 비교군 학습과 original-group 평가."""
import importlib,json,sys,time,math,collections
from pathlib import Path
import numpy as np
import torch
from .fp32_runtime import precision,policy,config,Guard,run
from .runtime import atomic_write_json as write
from .real_run import sha,record

def datasets(root):
    from .bf16_post import datasets as existing
    # 동일 policy의 신규 FP32 Page만 허용하며 기존 BF16 observation mapping은 쓰지 않습니다.
    p=policy(root)
    from .page import load_pages
    index=json.loads((root/'pages/index.json').read_text())
    for item in index.values():
        pg=load_pages(item['path'])[0]
        assert pg.provenance['policy_hash']==p['behavior_policy_id']
        assert pg.provenance['forward_dtype']=='float32' and pg.provenance['tf32'] is False
        assert pg.state.dtype==pg.updates.dtype==torch.float32
    ds=existing(root,behavior_dtype='float32')
    return ds

def train_worker(root,method,seed):
    precision();torch.set_num_threads(2);ds=datasets(root);cfg=config(root)
    cfg.run.run_id=f'{method}_seed{seed}';cfg.paths.output_root=str(root/'training')
    cfg.run.seeds.train=seed;cfg.run.seeds.data=cfg.run.seeds.split=cfg.run.seeds.sampler=cfg.run.seeds.eval=0
    cfg.data.source='pages';cfg.model.name=method
    cfg.train.task='joint' if method=='joint' else 'behavior'
    cfg.train.w_robust=0;cfg.train.w_pair_drop=1;cfg.train.w_max_drop=0;cfg.train.use_max_drop=False
    cfg.train.select_metric='L_pair_drop';cfg.train.microbatch_originals=1;cfg.eval.bootstrap_samples=2000
    cfg.train.batch_originals=8;cfg.train.max_epochs=100;cfg.train.patience=15
    assert cfg.train.lr==3e-4 and cfg.train.weight_decay==1e-3 and cfg.train.flow_weight==.1
    write(root/'training'/f'{method}_seed{seed}_config.json',cfg.to_dict())
    marker=cfg.run_dir/'fp32_completed.json'
    if marker.exists():
        assert json.loads(marker.read_text())['config_hash']==cfg.hash()
        return
    tm=importlib.import_module('aimo.train');original_build=tm.build_model;seen={'parameters':set(),'inputs':set(),'outputs':set(),'gradients':set()}
    def inspect_tensor(x,key):
        if isinstance(x,torch.Tensor) and x.is_floating_point():
            assert x.dtype==torch.float32,(key,x.dtype);seen[key].add(str(x.dtype))
        elif isinstance(x,(list,tuple)):
            for a in x:inspect_tensor(a,key)
        elif isinstance(x,dict):
            for a in x.values():inspect_tensor(a,key)
    def build(*args,**kwargs):
        model=original_build(*args,**kwargs)
        for p in model.parameters():
            inspect_tensor(p,'parameters')
            if p.requires_grad:p.register_hook(lambda grad:(inspect_tensor(grad,'gradients') or grad))
        for mod in model.modules():
            if isinstance(mod,(torch.nn.Linear,torch.nn.LayerNorm,torch.nn.MultiheadAttention)):
                mod.register_forward_pre_hook(lambda m,a:inspect_tensor(a,'inputs'))
                mod.register_forward_hook(lambda m,a,o:inspect_tensor(o,'outputs'))
        return model
    tm.build_model=build
    for name in ['_behavior_terms','_flow_terms']:
        original=getattr(tm,name)
        def checked(*a,_fn=original,**kw):
            result=_fn(*a,**kw)
            assert torch.isfinite(result.total).item(),'nonfinite loss'
            return result
        setattr(tm,name,checked)
    clip=torch.nn.utils.clip_grad_norm_
    def checked_clip(*a,**kw):kw['error_if_nonfinite']=True;return clip(*a,**kw)
    torch.nn.utils.clip_grad_norm_=checked_clip
    t=time.monotonic()
    summary=tm.train(cfg,ds,resume=(cfg.run_dir/'last.pt').exists(),guard=Guard(root))
    if Guard(root).should_stop()[0]:raise RuntimeError('external stop')
    assert summary['best_val_total'] is not None and math.isfinite(summary['best_val_total'])
    summary.update(wall_seconds=time.monotonic()-t,peak_vram_bytes=torch.cuda.max_memory_allocated())
    write(cfg.run_dir/'summary.json',summary)
    write(cfg.run_dir/'precision.json',{**precision(),'observed':{k:sorted(v) for k,v in seen.items()}})
    write(marker,{'config_hash':cfg.hash(),'finished':time.time()})

def baseline(ds,value):
    from .evaluate import MetricAccumulator
    from .losses import huber_elementwise,HUBER_DELTA
    acc=MetricAccumulator();hub=MetricAccumulator()
    for g in ds.groups:
        for lab in g.pair_labels.values():
            if lab.has_drop:
                err=value-lab.signed_drop;acc.add(g.original_id,abs(err))
                hub.add(g.original_id,float(huber_elementwise(torch.tensor(value),torch.tensor(lab.signed_drop),HUBER_DELTA)))
    return {'value':value,'metrics':{'pair_drop_mae':acc.summary(bootstrap_samples=2000,seed=0),
           'pair_drop_huber':hub.summary(bootstrap_samples=2000,seed=0)},
           'per_original_metrics':{'pair_drop_mae':{k:sum(v)/len(v) for k,v in acc.per_group.items()}},
           'robust_classification':None,'robust_head_status':'null/untrained'}

def compare(a,b):
    ids=sorted(set(a)&set(b))
    if not ids:return {'n_originals':0,'improvement':None,'ci95':None}
    d=np.array([a[i]-b[i] for i in ids],dtype=np.float64);rng=np.random.default_rng(0)
    boot=d[rng.integers(0,len(ids),(2000,len(ids)))].mean(1)
    return {'n_originals':len(ids),'improvement':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'direction':'positive means second model has smaller error','conditional_on_observed_four_slot_labels':True}

def evaluate(root):
    precision();torch.set_num_threads(2);ds=datasets(root)
    from .train import load_checkpoint,validation_metrics
    from .evaluate import evaluate_behavior,evaluate_dataset
    from .config import config_from_dict
    outputs={};outputs['zero']=baseline(ds['known_test'],0.)
    # train-only exact Huber constant: convex objective, deterministic 1-D derivative root.
    vals=[float(l.signed_drop) for g in ds['train'].groups for l in g.pair_labels.values() if l.has_drop]
    from .losses import HUBER_DELTA
    lo,hi=min(vals),max(vals)
    for _ in range(80):
        mid=(lo+hi)/2
        if sum(max(-HUBER_DELTA,min(HUBER_DELTA,mid-y)) for y in vals)>0:hi=mid
        else:lo=mid
    outputs['constant_fit']=baseline(ds['known_test'],(lo+hi)/2)
    frozen_hashes=None;hash_report={}
    for method,seeds in [('constant',[0]),('raw_change',[0]),('behavior_m0',[0]),('behavior',[0,1,2]),('joint',[0,1,2])]:
        for seed in seeds:
            name=f'{method}_seed{seed}';model,stats,payload=load_checkpoint(root/'training'/name/'best.pt')
            hashes=payload['hashes'];comparison={k:hashes[k] for k in ['split_hashes','train_subset_hash','label_policy_hash','data_seed','split_seed','stats_hash']}
            if frozen_hashes is None:frozen_hashes=comparison
            assert comparison==frozen_hashes,'data/label/split/normalization changed'
            hash_report[name]=comparison
            model=model.to('cuda');stats=stats.to('cuda')
            assert all(p.dtype==torch.float32 for p in model.parameters() if p.is_floating_point())
            result=evaluate_behavior(model,ds['known_test'],stats,bootstrap_samples=2000,seed=0,support_swap=True,microbatch=1)
            result['robust_classification']=None;result['robust_head_status']='null/untrained'
            result.pop('debug_untrained_robust_classification',None)
            outputs[name]=result;write(root/'evaluation'/f'{name}.json',result)
            if method=='joint' and seed==0:
                flow=evaluate_dataset(model,ds['known_test'],stats,horizons=(2,3,4),bootstrap_samples=2000,support_swap=True,seed=0)
                flow['exact_loss_terms']=validation_metrics(model,ds['known_test'],stats,config_from_dict(payload['config']))
                write(root/'evaluation/joint_flow.json',flow)
            del model,stats;torch.cuda.empty_cache()
    for name in ['zero','constant_fit']:write(root/'evaluation'/f'{name}.json',outputs[name])
    paired={}
    for seed in [0,1,2]:
        j=outputs[f'joint_seed{seed}']['per_original_metrics']['pair_drop_mae']
        paired[str(seed)]={other:compare(outputs[other]['per_original_metrics']['pair_drop_mae'],j) for other in ['zero','constant_fit','constant_seed0','raw_change_seed0','behavior_m0_seed0',f'behavior_seed{seed}']}
    variances={}
    for method in ['behavior','joint']:
        means=[np.mean(list(outputs[f'{method}_seed{s}']['per_original_metrics']['pair_drop_mae'].values())) for s in [0,1,2]]
        variances[method]={'seed_mae':means,'mean':float(np.mean(means)),'sample_sd':float(np.std(means,ddof=1))}
    write(root/'evaluation/hash_consistency.json',hash_report)
    write(root/'evaluation/completed.json',{'outputs':outputs,'paired':paired,'training_seed_variance':variances,
          'sampling_uncertainty':'four independent slots per prompt; bootstrap conditions on realized labels, not population correctness',
          'causal_or_official_robustness_claim':False})

def post(root):
    from .bf16_report import summarize
    summary=summarize(root,'collection')
    if (root/'runtime/oom_recovery_001').exists():
        from .fp32_recovery import counts,preserve_check
        preserve_check(root);counts(root)
    run(root,[(0,[sys.executable,'-m','aimo.fp32_collect','pages',str(root)])],'pages')
    ds=datasets(root)
    valid={s:sum(any(l.has_drop for l in g.pair_labels.values()) for g in d.groups) for s,d in ds.items()}
    write(root/'labels/usable_originals.json',{'counts':valid,'historical_minimum_counts_reporting_only':{'train':24,'validation':8,'known_test':12}})
    if not valid.get('train') or not valid.get('validation'):
        write(root/'runtime/completed.json',{'status':'DATA_LIMIT','training':False,'valid':valid})
        finalize(root);return
    del ds
    jobs=[(0,'constant,raw_change'),(1,'behavior_m0'),(2,'behavior'),(3,'joint')]
    run(root,[(gpu,[sys.executable,'-m','aimo.fp32_post','train',str(root),methods,'0']) for gpu,methods in jobs],'training')
    run(root,[(gpu,[sys.executable,'-m','aimo.fp32_post','train',str(root),method,'1,2']) for gpu,method in [(0,'behavior'),(1,'joint')]],'training_seeds')
    run(root,[(0,[sys.executable,'-m','aimo.fp32_post','evaluate',str(root)])],'evaluation')
    write(root/'runtime/completed.json',{'status':'EVALUATION_COMPLETED','training':True})
    finalize(root)

def finalize(root):
    collection=json.loads((root/'collection/summary.json').read_text())
    completion=json.loads((root/'runtime/completed.json').read_text())
    detail={'collection':collection,'completion':completion,
       'precision_manifest':json.loads((root/'protocol/precision_manifest.json').read_text()),
       'execution_amendment':json.loads((root/'protocol/execution_amendment.json').read_text()),
       'code':json.loads((root/'protocol/executed_code.json').read_text()),
       'current_cost':json.loads((root/'gpu_budget.json').read_text()),
       'resume':str(root/'runtime/resume.sh'),'pages':json.loads((root/'pages/summary.json').read_text())}
    if (root/'evaluation/completed.json').exists():
        ev=json.loads((root/'evaluation/completed.json').read_text())
        detail['model_mae']={k:v['metrics']['pair_drop_mae'] for k,v in ev['outputs'].items()}
        detail['paired']=ev['paired'];detail['training_seed_variance']=ev['training_seed_variance']
        detail['support_swap']={k:v.get('metrics',{}).get('pair_support_swap_mae_increase') for k,v in ev['outputs'].items() if k.startswith('joint')}
    if (root/'runtime/page_parallel_cost.json').exists():
        auxiliary=json.loads((root/'runtime/page_parallel_cost.json').read_text())
        detail['parallel_page_cost']=auxiliary
        detail['current_cost']['gpu_hours']+=auxiliary['gpu_hours']
        detail['current_cost']['parallel_page_gpu_hours_included']=True
    if (root/'runtime/oom_recovery_001').exists():
        from .fp32_recovery import counts,preserve_check
        preserve_check(root)
        a=root/'runtime/oom_recovery_001'
        detail['oom_recovery']={
            'diagnosis':json.loads((a/'oom_diagnosis.json').read_text()),
            'settings':json.loads((root/'runtime/memory_settings.json').read_text()),
            'smoke':json.loads((a/'smoke_passed.json').read_text()),
            'final_operational_summary':counts(root),
            'code':json.loads((a/'code_manifest.json').read_text())}
    detail['interpretation']='관측된 네 번 sampling의 signed drop 예측. 인과 메커니즘 또는 공식 robustness 검증이 아님.'
    write(root/'final_summary.json',detail);record(root,completion['status'],detail)
    from .fp32_report import report
    report(root,detail)

if __name__=='__main__':
    mode=sys.argv[1];root=Path(sys.argv[2])
    if mode=='train':
        for method in sys.argv[3].split(','):
            for seed in map(int,sys.argv[4].split(',')):
                train_worker(root,method,seed)
                __import__('gc').collect();torch.cuda.empty_cache()
    elif mode=='evaluate':evaluate(root)
    elif mode=='post':post(root)
    elif mode=='finalize':finalize(root)
