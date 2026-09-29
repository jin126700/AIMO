"""FP32 primary 실행: precision 고정, 유한 작업, durable 재개와 자원 기록."""
import dataclasses, json, os, sys, time, subprocess, fcntl, shutil
from pathlib import Path
import torch
from .real_run import cfg_for, sha, REV
from .runtime import atomic_write_json as write
PARENT = Path('/data1/HKM/result/AIMO/deepmath_bf16_bucket_20260926T210413Z')

def precision():
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.is_autocast_enabled()
    assert not torch.is_autocast_enabled('cpu')
    return {'default_dtype':str(torch.get_default_dtype()), 'autocast_cuda':False,
            'autocast_cpu':False, 'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn':torch.backends.cudnn.allow_tf32, 'precision_api':'PyTorch 2.7 allow_tf32',
            'torch':torch.__version__, 'cuda':torch.version.cuda, 'amp':False, 'fallback':False}

def memory_settings(root):
    path=Path(root)/'runtime/memory_settings.json'
    if not path.exists():
        return {'batch_size':4,'allocator_fraction':.9,'replicas':4,'execution_id':'initial_batch4'}
    settings=json.loads(path.read_text())
    assert settings['batch_size']==1 and settings['replicas']==4
    assert settings['policy_hash']==policy(root)['behavior_policy_id']
    assert settings['dtype']=='float32' and settings['max_new_tokens']==16384
    assert settings['max_total_context']==40960
    return settings

def config(root):
    cfg=cfg_for(root); cfg.server.thinking.samples_per_prompt=4
    assert cfg.server.thinking.numerical_backend=='fp32_sdpa'
    return cfg

def policy(root):
    p=json.loads((root/'protocol/policy.json').read_text())
    assert p['profile']['numerical_backend']=='fp32_sdpa'
    assert p['precision']['tf32_matmul'] is False and p['precision']['tf32_cudnn'] is False
    assert p['cohort_id']==root.name and p['primary_dtype']=='float32'
    return p

class Guard:
    def __init__(self,root,*ignored): self.root=Path(root)
    def should_stop(self):
        return (self.root/'STOP').exists(), 'external_stop'
    def elapsed(self):
        return json.loads((self.root/'gpu_budget.json').read_text())['active_seconds']

def prepare(root):
    import transformers
    if (root/'protocol/manifest.json').exists():raise RuntimeError('Frozen run already prepared; use resume')
    from .adapters.qwen import SCORER_VERSION
    from transformers import AutoTokenizer
    cfg=config(root); tok=AutoTokenizer.from_pretrained(cfg.server.thinking.model_path,local_files_only=True)
    p={'profile':dataclasses.asdict(cfg.server.thinking),'backend':'native_dynamiccache_v1',
       'transformers':transformers.__version__,'precision':precision(),'primary_dtype':'float32',
       'cohort_id':root.name,'rng':'independent_cuda_generator_per_slot','max_batch':4,
       'replicas':4,'attention':'sdpa','use_cache':True,'output_hidden_states':False,
       'output_attentions':False,'output_scores':False,'scheduler':'frozen_parent_order_stride4'}
    pid=sha(json.dumps(p,sort_keys=True))
    obs={'source':'qwen3_real','policy_hash':pid,'model_hash':REV,
         'tokenizer_hash':sha(tok.backend_tokenizer.to_str()),'template_hash':sha(tok.chat_template),
         'forward_dtype':'float32','tf32':False,'autocast':False}
    p.update(behavior_policy_id=pid,observation_policy_id=pid,observation_provenance=obs,
             target_model_revision=REV,parent_run=str(PARENT))
    write(root/'protocol/policy.json',p)
    for stage,expected in [('calibration',36),('collection',624)]:
        groups=json.loads((PARENT/stage/'groups.json').read_text())
        for g in groups:
            for pr in g['prompts']:
                ids=tok.apply_chat_template([{'role':'user','content':pr['prompt']}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)
                assert ids==pr['input_ids'] and sha(json.dumps(ids))==pr['input_token_hash']
                assert len(ids)+16384<=cfg.server.thinking.max_total_context
                for i,s in enumerate(pr['slots']):
                    sid=f'{root.name}:{pid}:{stage}:{pr["prompt_id"]}:{i}'
                    s.update(slot_id=sid,seed=int(sha(sid)[:8],16))
        assert sum(len(pr['slots']) for g in groups for pr in g['prompts'])==expected
        assert len({s['slot_id'] for g in groups for pr in g['prompts'] for s in pr['slots']})==expected
        write(root/stage/'groups.json',groups)
    prior=json.loads((PARENT/'gpu_budget.json').read_text())
    write(root/'gpu_budget.json',{'active_seconds':0.,'gpu_hours':0.,'active_workers':0,'updated_at':time.time()})
    write(root/'protocol/execution_amendment.json',{
        'time_limits_minutes':None,'removed_cutoffs':[120,150,175,180],
        'five_hours_is_not_a_cap':True,'sampling_changed':False,'max_new_tokens':16384,
        'finite_slots':{'calibration':36,'collection':624},'train_seeds':{'behavior':[0,1,2],'joint':[0,1,2]},
        'minimum_counts_reporting_only':True,'no_external_automation':True,
        'prior_cost_authoritative_ledger':str(PARENT/'gpu_budget.json'),'prior_cost':prior,
        'prior_cost_includes_earlier_FP32_and_BF16':True,'prior_BF16_results_seen':True,
        'untouched_dataset_claim':False,'old_pages_not_relabelled':True,
        'page_reuse_decision':'fresh extraction: old TF32 evidence insufficient',
        'calibration_reuse_decision':'new 36 slots: old backend/precision evidence not identical'})
    manifest={'remote_commit':(root/'protocol/code_sha.txt').read_text().strip(),
              'before_patch_sha256':sha((root/'protocol/before.patch').read_bytes()),
              'before_source_sha256':sha((root/'protocol/before_source.tar.gz').read_bytes()),
              'frozen_sha256':sha((root/'protocol/frozen.json').read_bytes()),
              'parent_frozen_sha256':sha((PARENT/'protocol/frozen.json').read_bytes()),
              'policy_hash':pid,'plan_hashes':{s:sha((root/s/'groups.json').read_bytes()) for s in ['calibration','collection']},
              'model_files_manifest_reference':str(PARENT.parent/'deepmath_real_20260926T193418Z/protocol/model.sha256')}
    assert manifest['frozen_sha256']==manifest['parent_frozen_sha256']
    write(root/'protocol/manifest.json',manifest)

def run(root,commands,stage):
    verify(root)
    (root/stage).mkdir(exist_ok=True)
    with (root/'runtime/supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=json.loads((root/'gpu_budget.json').read_text()); start=last=time.monotonic()
        gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free','--format=csv'],text=True)
        gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
        write(root/stage/'gpu_preflight.json',{'time':time.time(),'gpus':gpu_state,'processes':gpu_processes})
        hours=old['gpu_hours']; children=[]; logs=[]
        try:
            for gpu,command in commands:
                log=(root/stage/f'gpu{gpu}.log').open('a'); logs.append(log)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
                children.append((gpu,subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)))
            while True:
                now=time.monotonic();alive=[(g,p) for g,p in children if p.poll() is None]
                hours+=(now-last)*len(alive)/3600;last=now
                write(root/'gpu_budget.json',{'active_seconds':old['active_seconds']+now-start,'gpu_hours':hours,
                      'active_workers':len(alive),'updated_at':time.time(),'owned_pids':[p.pid for _,p in alive],'stage':stage})
                if not alive:break
                failures=[p.returncode for _,p in children if p.poll() is not None and p.returncode!=0]
                if stage in ('collection','calibration','memory_smoke') and now-start>3600:
                    evidence=list((root/stage).rglob('*.json'))+list((root/'runtime').glob('progress_*.json'))
                    newest=max([p.stat().st_mtime for p in evidence]+[time.time()-(now-start)])
                    if time.time()-newest>3600:failures.append('no durable generation progress for 60 minutes')
                if failures or shutil.disk_usage(root).free<10*1024**3:
                    (root/'STOP').write_text('worker failure or disk below 10GiB; preserve and inspect')
                time.sleep(5)
        finally:
            for _,p in children:
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
            for log in logs:log.close()
            write(root/'gpu_budget.json',{'active_seconds':old['active_seconds']+time.monotonic()-start,
                  'gpu_hours':hours,'active_workers':0,'updated_at':time.time(),'owned_pids':[],'stage':stage})
        codes=[p.returncode for _,p in children]
        write(root/stage/'exit.json',{'codes':codes,'stage_seconds':time.monotonic()-start})
        if any(c!=0 for c in codes) or Guard(root).should_stop()[0]:raise RuntimeError(f'{stage} stopped: {codes}')
        return codes

def verify(root):
    p=policy(root)
    manifest=json.loads((root/'protocol/manifest.json').read_text())
    assert manifest['policy_hash']==p['behavior_policy_id']
    assert sha((root/'protocol/frozen.json').read_bytes())==manifest['frozen_sha256']
    for stage,digest in manifest['plan_hashes'].items():
        assert sha((root/stage/'groups.json').read_bytes())==digest,'frozen request manifest changed'
    if (root/'protocol/policy.sha256').exists():
        assert sha((root/'protocol/policy.json').read_bytes())==(root/'protocol/policy.sha256').read_text().strip()
    return True

def snapshot(root):
    paths=[str(p) for d in ['src','configs','tests'] for p in Path(d).rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc']
    write(root/'protocol/executed_source_hashes.json',{p:sha(Path(p).read_bytes()) for p in paths if Path(p).is_file()})
    patch=(root/'protocol/executed.patch').read_bytes()
    write(root/'protocol/executed_code.json',{'sha':(root/'protocol/code_sha.txt').read_text().strip(),
          'dirty_patch_sha256':sha(patch),'source_hashes_sha256':sha((root/'protocol/executed_source_hashes.json').read_bytes())})

if __name__=='__main__':
    root=Path(sys.argv[2])
    if sys.argv[1]=='prepare':prepare(root)
    elif sys.argv[1]=='snapshot':snapshot(root)
    elif sys.argv[1]=='verify':verify(root)


    elif sys.argv[1]=='stage':
        stage=sys.argv[3]
        run(root,[(i,[sys.executable,'-m','aimo.fp32_collect','worker',str(root),stage,str(i)]) for i in range(4)],stage)
