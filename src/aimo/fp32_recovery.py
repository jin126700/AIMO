"""동일 FP32 cohort의 OOM-safe smoke, 보존 검증, 전체 파이프라인 재개."""
import json,sys,time,fcntl,traceback,collections
from pathlib import Path
import numpy as np
from .runtime import atomic_write_json as write
from .fp32_runtime import run,verify,memory_settings,Guard
from .real_run import sha,record
from .bf16_collect import slot_path

def preserve_check(root):
    a=root/'runtime/oom_recovery_001'
    for name,digest in json.loads((a/'preserved_hashes.json').read_text()).items():
        p=Path(name) if Path(name).is_absolute() else root/name
        assert sha(p.read_bytes())==digest, f'protected evidence changed: {name}'
    assert (root/'collection/slots.jsonl').read_bytes().startswith((a/'collection_ledger_before.jsonl').read_bytes()),'old ledger changed'
    assert json.loads((root/'runtime/memory_settings.json').read_text())==json.loads((a/'memory_settings_frozen.json').read_text())
    verify(root)
    return True

def counts(root):
    groups=json.loads((root/'collection/groups.json').read_text())
    counts=collections.Counter();resolved=0;original_distribution=collections.Counter()
    for g in groups:
        for i,p in enumerate(g['prompts']):
            outcomes=[]
            for s in p['slots']:
                path=slot_path(root,'collection',s['slot_id'])
                if not path.exists():code='not_started'
                else:
                    raw=json.loads(path.read_text())
                    code=raw.get('outcome','U_score') if raw['state']=='completed' else 'interrupted'
                counts[code]+=1;outcomes.append(code)
            if all(c in ['C','W'] for c in outcomes):
                resolved+=1
                if i==0:original_distribution[str(outcomes.count('C')/4)]+=1
    labels=json.loads((root/'labels/collection_pairs.json').read_text()) if (root/'labels/collection_pairs.json').exists() else []
    valid=[l['signed_drop'] for l in labels if l['signed_drop'] is not None]
    summary={'counts':{k:counts[k] for k in ['C','W','X','U_score','interrupted','not_started']},
             'resolved_prompts':resolved,'planned_prompts':156,'valid_pairs':len(valid),
             'drop_zero':sum(v==0 for v in valid),'drop_positive':sum(v>0 for v in valid),
             'drop_negative':sum(v<0 for v in valid),'original_correctness_distribution':dict(original_distribution),
             'interrupted_replaced':False,'binary_robust_label':None}
    write(root/'collection/operational_summary.json',summary)
    return summary

def check_smoke(root):
    a=root/'runtime/oom_recovery_001';plan=json.loads((a/'smoke_plan.json').read_text())
    rows=[];batches=[]
    for sid in plan['slot_ids']:
        raw=json.loads(slot_path(root,'collection',sid).read_text())
        assert raw['state']=='completed' and raw['termination'] in ('eos','token_cap')
        assert raw['identity']['policy_hash']==memory_settings(root)['policy_hash']
        rows.append(raw)
        b=json.loads((root/'runtime/memory_batches'/(sha(sid)+'.json')).read_text())
        assert b['batch_size']==1 and b['kv_dtypes']==['torch.float32']
        assert b['logits_dtype']=='torch.float32'
        assert b['after_allocated_bytes']<=b['before_allocated_bytes']+64*1024**2
        assert b['peak_vram_bytes']<32*1024**3 and b['peak_reserved_bytes']<35*1024**3
        batches.append(b)
    assert len(batches)>=3
    after=[b['after_allocated_bytes'] for b in batches]
    assert max(after)-min(after)<64*1024**2,'persistent allocation accumulated'
    stats={'passed':True,'slots':len(rows),'consecutive_batches':len(batches),
           'peak_allocated_bytes':max(b['peak_vram_bytes'] for b in batches),
           'peak_reserved_bytes':max(b['peak_reserved_bytes'] for b in batches),
           'max_kv_cache_bytes':max(b['max_cache_bytes'] for b in batches),
           'after_allocated_min':min(after),'after_allocated_max':max(after),
           'generated_tokens':sum(b['generated_tokens'] for b in batches),
           'batch_wall_seconds':sum(b['wall_seconds'] for b in batches),
           'outcomes':dict(collections.Counter(r['outcome'] for r in rows)),
           'token_cap':16384,'dtype':'float32','time_limit':None}
    stats['single_replica_tokens_per_second']=stats['generated_tokens']/stats['batch_wall_seconds']
    preserve_check(root);write(a/'smoke_passed.json',stats)
    return stats

def eta(root,stats):
    groups=json.loads((root/'collection/groups.json').read_text())
    pending=[sum(not slot_path(root,'collection',s['slot_id']).exists() for g in groups[i::4] for p in g['prompts'] for s in p['slots']) for i in range(4)]
    per=stats['batch_wall_seconds']/stats['slots']
    nominal=max(pending)*per/3600
    value={'remaining_slots':sum(pending),'pending_by_replica':pending,'replicas':4,'batch_size':1,
           'observed_single_replica_tokens_per_second':stats['single_replica_tokens_per_second'],
           'observed_seconds_per_slot':per,'collection_hours_nominal':nominal,
           'collection_hours_low':nominal*.7,'collection_hours_high':nominal*1.6,
           'basis':'same FP32 batch1 actual frozen smoke group; fixed 4-replica queue, length variation and group imbalance',
           'training_evaluation_eta':'not reliable until first actual training epoch; excluded from numeric collection range',
           'runtime_limit':None,'necessary':'remaining frozen FP32 slots cannot be replaced with BF16 or selected successes'}
    write(root/'runtime/eta.json',value)
    return value

def main(root):
    a=root/'runtime/oom_recovery_001'
    with (root/'runtime/pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preserve_check(root)
        if Guard(root).should_stop()[0]:raise RuntimeError('STOP still active; inspect before resuming')
        if not (a/'smoke_passed.json').exists():
            record(root,'OOM_RECOVERY_MEMORY_SMOKE',{'memory_settings':memory_settings(root),'smoke_slots':12,
                   'preserved':{'completed':26,'interrupted':10,'not_started_before_smoke':588},
                   'scientific_failure':False,'eta':'pending stable FP32 smoke throughput'})
            run(root,[(2,[sys.executable,'-m','aimo.fp32_collect','smoke',str(root)])],'memory_smoke')
            stats=check_smoke(root)
        else:stats=json.loads((a/'smoke_passed.json').read_text())
        estimate=eta(root,stats)
        record(root,'OOM_SAFE_COLLECTION_RESUMING',{'smoke':stats,'eta':estimate,'preserved_original_evidence':True})
        print(json.dumps({'smoke':stats,'eta':estimate}),flush=True)
        from .fp32_pipeline import main as pipeline
        pipeline(root)
        preserve_check(root)
        counts(root)

if __name__=='__main__':
    r=Path(sys.argv[1])
    try:main(r)
    except BaseException as exc:
        write(r/'runtime/failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'time':time.time(),
              'resume_entry':'aimo.fp32_recovery','old_evidence_preserved':True})
        raise
