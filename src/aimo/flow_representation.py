"""E-FLOW-1: label-free Flow 학습과 frozen representation 실험."""
from __future__ import annotations
import hashlib,json,time,shutil,traceback,fcntl
from pathlib import Path
import numpy as np
import torch
from .runtime import atomic_write_json as write
from .data import OriginalGroup,PageDataset
from .model import LoopedPredictor,PersistencePredictor
SCHEMA="flow-observed-mean-v1"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def split_ids(ids):
    ordered=sorted(ids,key=lambda x:hashlib.sha256(('E-FLOW-1:'+x).encode()).hexdigest())
    n=max(1,round(len(ordered)/8))
    if len(ordered)<3:raise ValueError("at least three train originals")
    return ordered[n:],ordered[:n]

def load_manifest(manifest, ids):
    """Page-only allowlist loader. Outcome/label paths are not accepted."""
    from .page import load_pages
    groups=[]
    for ident in ids:
        row=manifest['originals'][ident];pages=[]
        for item in row['pages']:
            assert digest(item['path'])==item['sha256']
            p=load_pages(item['path'])[0]
            assert p.original_id==ident and p.state.dtype==p.updates.dtype==torch.float32
            pages.append(p)
        groups.append(OriginalGroup(ident,pages[0],pages[1:],ident))
    return PageDataset('pages',groups)

def assert_label_free(ds):
    for d in ds.values():
        for g in d.groups:
            if g.pair_labels or g.panel_label is not None:
                raise ValueError("Stage1 forbids behavior labels")

def stage1_train(cfg,ds):
    from .train import train
    assert_label_free(ds)
    assert set(ds)=={'train','validation'}
    assert cfg.train.task=='flow' and cfg.train.resolved_select_metric()=='flow_total'
    assert cfg.model.name=='loop4'
    assert not ({g.original_id for g in ds['train'].groups}&{g.original_id for g in ds['validation'].groups})
    result=train(cfg,ds,resume=(cfg.run_dir/'last.pt').exists())
    result['behavior_supervision_seen']=False
    result['task']='flow_representation'
    return result

def prepare(source,root):
    from .config import load_config
    root.mkdir(parents=True,exist_ok=True)
    groups=json.loads((source/'collection/groups.json').read_text())
    index=json.loads((source/'pages/index.json').read_text())
    policy=json.loads((source/'protocol/policy.json').read_text())
    manifest={'source':str(source),'originals':{},'policy_hash':policy['behavior_policy_id']}
    from .page import load_pages
    for g in groups:
        entries=[]
        for pr in g['prompts']:
            item=index[pr['prompt_id']]
            assert digest(item['path'])==item['sha256']
            p=load_pages(item['path'])[0]
            assert p.provenance['policy_hash']==manifest['policy_hash']
            assert p.provenance['forward_dtype']=='float32' and p.provenance['tf32'] is False
            assert p.state.dtype==p.updates.dtype==torch.float32
            entries.append({k:item[k] for k in ['path','sha256']})
        manifest['originals'][g['original_id']]={'split':g['split'],'pages':entries}
    train_ids=[i for i,r in manifest['originals'].items() if r['split']=='train']
    ft,fd=split_ids(train_ids)
    splits={'flow_train':ft,'flow_dev':fd,'train':train_ids,
            'validation':[i for i,r in manifest['originals'].items() if r['split']=='validation'],
            'known_test':[i for i,r in manifest['originals'].items() if r['split']=='known_original_test']}
    cfg=load_config('configs/stage1.yaml')
    cfg.run.run_id='flow';cfg.run.device='cuda';cfg.paths.output_root=str(root)
    cfg.train.task='flow';cfg.train.select_metric='flow_total'
    cfg.train.max_epochs=100;cfg.train.patience=15;cfg.eval.bootstrap_samples=2000
    cfg.train.w_next=1.;cfg.train.w_within=1.;cfg.train.w_roll=.25
    cfg.train.w_robust=cfg.train.w_pair_drop=cfg.train.w_max_drop=0.
    spec={'task':'flow_representation','stage1_config':cfg.to_dict(),'source':str(source),
          'representation_schema':SCHEMA,'representation_dim':128,'panel_dim':256,
          'criterion':None,'criterion_status':'NO_VERIFIED_REAL_ROBUSTNESS_DEFINITION',
          'criterion_evidence':'source config robust_policy.enabled=false, definition_id/source=null; threshold 0.25 belongs to synthetic toy only',
          'probe_C':[.01,.1,1.],'shuffle_repeats':20,'bootstrap':2000,
          'flow_gate':'flow_dev aggregate L_flow at least 5% below both zero and train_mean; diagnostic threshold frozen before training',
          'no_qwen_generation':True,'checkpoint_selection':'flow_dev L_flow only'}
    for name,value in [('page_manifest.json',manifest),('split_manifest.json',splits),('config.json',spec)]:
        p=root/name
        if p.exists():assert json.loads(p.read_text())==value
        else:write(p,value)
    write(root/'source_hashes.json',{str(p.relative_to(source)):digest(p) for p in [source/'pages/index.json',source/'collection/groups.json',source/'labels/collection_pairs.json',source/'protocol/policy.json']})
    return spec,manifest,splits

class MeanFlow(torch.nn.Module):
    """flow_train original-balanced depth/channel/native-coordinate mean; no labels."""
    def __init__(self,ds):
        super().__init__()
        rows=[]
        for g in ds.groups:
            a=[]
            for v in g.variants:
                mask=g.original.valid&v.valid
                a.append((v.updates-g.original.updates)[:,mask].mean(1))
            rows.append(torch.stack(a).mean(0))
        self.register_buffer('mean',torch.stack(rows).mean(0))
    def forward(self,inp,stats):
        value=self.mean[inp.cut]/stats.target_scale[inp.cut].clamp_min(stats.floor)[:,None]
        return value[None,None].expand(inp.batch_size,inp.valid.shape[1],-1,-1)

def flow_evaluate(model,ds,stats,cfg):
    from .train import validation_metrics
    result={}
    for g in ds.groups:
        result[g.original_id]=validation_metrics(model,PageDataset(ds.split,[g]),stats,cfg)
    keys=['L_next','L_within','L_roll','flow_total']
    means={k:float(np.mean([v[k] for v in result.values() if np.isfinite(v.get(k,np.nan))])) for k in keys}
    return {'mean':means,'per_original':result,'n_originals':len(result),'aggregation':'original-balanced'}

def extract(model,stats,ds):
    model.eval().requires_grad_(False)
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    pairs=[];panels=[]
    from .flow_probe import panel_pool
    for split,d in ds.items():
        for g in d.groups:
            z=[];raw=[];ids=[]
            for v in g.variants:
                if not bool((v.valid&g.original.valid).any()):continue
                a=model.encode_pair(g.original,v,stats).squeeze(0).cpu().numpy()
                z.append(a);ids.append(v.variant_id)
                valid=v.valid&g.original.valid
                raw.append(float(torch.cat([(v.state-g.original.state)[:,valid].flatten(),
                    (v.updates-g.original.updates)[:,valid].flatten()]).square().mean().sqrt()))
                pairs.append({'original_id':g.original_id,'variant_id':v.variant_id,'split':split,'z':a.tolist()})
            if not z:continue
            m0=model.encode_pair(g.original,g.original,stats,original_only=True).squeeze(0).cpu().numpy()
            panels.append({'original_id':g.original_id,'split':split,'z':panel_pool(np.asarray(z)).tolist(),
                           'm0':m0.tolist(),'raw':[float(np.mean(raw)),float(np.max(raw))],
                           'coverage':{'expected':len(g.variants),'actual':len(z),'members':ids},
                           'dispersion':float(np.mean(np.sum((np.asarray(z)-np.mean(z,axis=0))**2,axis=1)))})
    assert all(torch.equal(before[k],v.cpu()) for k,v in model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return pairs,panels

def run(root):
    start=time.monotonic()
    from .config import config_from_dict
    from .train import load_checkpoint
    from .fp32_runtime import precision
    precision();torch.set_num_threads(2)
    spec=json.loads((root/'config.json').read_text());manifest=json.loads((root/'page_manifest.json').read_text())
    splits=json.loads((root/'split_manifest.json').read_text());cfg=config_from_dict(spec['stage1_config'])
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'status.json',{'stage':'flow_training','started':time.time()})
        ds={'train':load_manifest(manifest,splits['flow_train']),
            'validation':load_manifest(manifest,splits['flow_dev'])}
        if not (root/'flow_train_summary.json').exists():
            summary=stage1_train(cfg,ds);write(root/'flow_train_summary.json',summary)
            shutil.copyfile(cfg.run_dir/'best.pt',root/'best_flow.pt')
        model,stats,payload=load_checkpoint(root/'best_flow.pt')
        model=model.cuda();stats=stats.to('cuda')
        write(root/'status.json',{'stage':'flow_validation','time':time.time()})
        if not (root/'flow_eval_summary.json').exists():
            controls={'zero':PersistencePredictor(ds['train'].hidden_size,ds['train'].n_landmarks).cuda(),
                      'train_mean':MeanFlow(ds['train']).cuda(),'flow':model}
            results={k:flow_evaluate(m,ds['validation'],stats,cfg) for k,m in controls.items()}
            value=results['flow']['mean']['flow_total']
            established=all(value<.95*results[k]['mean']['flow_total'] for k in ['zero','train_mean'])
            results['verdict']='FLOW_REPRESENTATION_ESTABLISHED' if established else 'FLOW_REPRESENTATION_NOT_ESTABLISHED'
            write(root/'flow_eval_summary.json',results)
        del ds
        write(root/'status.json',{'stage':'frozen_features','time':time.time()})
        all_ds={k:load_manifest(manifest,splits[k]) for k in ['train','validation','known_test']}
        pairs,panels=extract(model,stats,all_ds)
        from .model import build_model
        torch.manual_seed(0)
        random=build_model(cfg,all_ds['train'].hidden_size,all_ds['train'].n_blocks,all_ds['train'].n_landmarks).cuda()
        rp,rc=extract(random,stats,all_ds)
        write(root/'pair_embeddings.json',pairs);write(root/'panel_features.json',panels)
        write(root/'random_pair_embeddings.json',rp);write(root/'random_panel_features.json',rc)
        write(root/'representation_manifest.json',{'schema':SCHEMA,'checkpoint_sha256':digest(root/'best_flow.pt'),
              'page_manifest_sha256':digest(root/'page_manifest.json'),'pair_embeddings_sha256':digest(root/'pair_embeddings.json'),
              'panel_features_sha256':digest(root/'panel_features.json'),'dim':128,'panel_dim':256,
              'encoder_frozen':True,'random_checkpoint_loaded':False,'normalization':'flow_train only; shared with random control'})
        from .flow_probe import finish
        finish(root,spec,panels,rc,pairs)
        source=Path(spec['source'])
        for name,h in json.loads((root/'source_hashes.json').read_text()).items():assert digest(source/name)==h
        for row in manifest['originals'].values():
            for p in row['pages']:assert digest(p['path'])==p['sha256']
        write(root/'completed.json',{'status':'COMPLETED','seconds':time.monotonic()-start,
              'source_preserved':True,'no_behavior_supervision_in_stage1':True})
        report(root)

def report(root):
    spec=json.loads((root/'config.json').read_text());f=json.loads((root/'flow_eval_summary.json').read_text())
    p=json.loads((root/'probe_results.json').read_text());labs=json.loads((root/'robustness_labels.json').read_text())
    summary=json.loads((root/'flow_train_summary.json').read_text())
    lines=['# E-FLOW-1','',
      '직전 behavior regression에서 81 valid pairs 중 77 pair가 zero-drop이었고, zero baseline이 Joint보다 우수하여, pair-drop direct supervision을 primary에서 제거하고 Flow representation learning + frozen robustness probe로 전환.',
      '',f"Flow 판정: {f['verdict']}",f"Probe 판정: {p['status']}",
      '81 pairs는 전체 split 합계이며 이전 test는 19 pairs / 10 originals.',
      '', '## Architecture 및 protocol',
      'Shared LoopedCore ×4, 128/4 heads/256 FFN/dropout0.1. L_next + L_within + 0.25 L_roll.',
      'Stage1 Page-only loader, flow_train 28 / flow_dev 4, ID hash split. 외부 validation/test는 학습 및 checkpoint 선택에 미사용.',
      'behavior_supervision_seen = false. FP32, AMP/autocast/TF32 OFF. 새 Qwen inference 없음.',
      'z: full observed valid cells mask-mean128; panel mean/std256, std ddof0. 새로운 head/query 없음.',
      '', '## Flow dev 결과 (original-balanced)',
      '| model | next | within | rollout2/4 | total |','|---|---:|---:|---:|---:|']
    for k in ['zero','train_mean','flow']:
        x=f[k]['mean'];lines.append(f"| {k} | {x['L_next']:.6g} | {x['L_within']:.6g} | {x['L_roll']:.6g} | {x['flow_total']:.6g} |")
    lines+=['',f"학습 epochs: {len(summary['history'])}; best epoch: {summary.get('best_epoch')}",
       '', '## Robustness criterion / controls',json.dumps(labs['counts'],ensure_ascii=False),
       '실데이터 robust_policy가 disabled이고 검증된 binary threshold가 없어 primary robust label을 만들지 않았다. toy threshold를 이전하지 않았다.',
       'Flow/M0/RawChange/Random feature 추출은 완료. BCE, 20회 label-shuffle null, classification bootstrap은 label 부재로 미실행이며 null 상태를 artifact에 명시.',
       'Panel max-drop bounds와 unresolved coverage는 보존. pair label을 panel label로 복사하지 않음.',
       '', '## 판단과 한계',
       'Flow decodability ≠ robustness; probe separability ≠ causal mechanism. 기존 test를 본 이후의 exploratory 연구다.',
       '다음: Flow gate 실패 시 Page/Flow task 검토. 통과해도 검증된 robustness criterion과 outcome-blind behavior variation 설계가 먼저이며 causal analysis로 진행하지 않음.',
       'PCA는 train에서 fit한 diagnostic only. 클래스 부재로 centroid distance는 정의되지 않음.',
       '',f"Artifacts: {root}",f"Reproduce: python -m aimo flow-representation-experiment --run-dir {root} --execute-gpu",
       f"Runtime: {json.loads((root/'completed.json').read_text())['seconds']:.1f} seconds",
       'Code/provenance: provenance/head.txt, before.patch, implementation.patch, source_hashes.json']
    text='\n'.join(lines)+'\n';(root/'report.md').write_text(text)
    log=Path('/data1/HKM/AIMO/Loop_result.md')
    marker=f'## {root.name}'
    if marker not in log.read_text():
        with log.open('a') as stream:stream.write('\n'+marker+'\n\n'+text)

def command(args):
    root=Path(args.run_dir)
    if getattr(args,'source',None):prepare(Path(args.source),root)
    if not args.execute_gpu:return 0
    try:run(root)
    except BaseException:
        write(root/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
    return 0
