"""Owned subprocess supervisor with one shared active-wall-time budget."""
import collections
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from .real_run import cfg_for,sha,record,REV
from .runtime import atomic_write_json,DedupLedger

class ReadBudget:
    def __init__(self,root,cutoff=120):self.root=root;self.cutoff=cutoff
    def elapsed(self):
        d=json.loads((self.root/'gpu_budget.json').read_text())
        return d['active_seconds']+max(0,time.time()-d.get('updated_at',time.time()))
    def should_stop(self):
        stop=(self.root/'STOP').exists() or self.elapsed()>=self.cutoff*60
        return stop,'shared_budget_or_external_stop' if stop else ''

def worker(root,index,workers,stage):
    from .adapters.qwen import load_real_qwen,QwenGenerationBackend
    from .collect import CollectionPlan,collect_outcomes
    torch.set_num_threads(2)
    cfg=cfg_for(root)
    if stage=='collection':cfg.server.thinking.samples_per_prompt=4
    plans=[CollectionPlan(**{k:v for k,v in p.items() if k in CollectionPlan.__dataclass_fields__}) for p in json.loads((root/stage/'requests.json').read_text())]
    selected=plans[index::workers]
    budget=ReadBudget(root)
    if budget.should_stop()[0]:return
    model,tok=load_real_qwen(cfg);backend=QwenGenerationBackend(cfg.server.thinking,model,tok)
    report=[];times=[]
    for plan in selected:
        t=time.monotonic()
        r=collect_outcomes(backend,[plan],cfg.server.thinking,ledger=DedupLedger.open(root/stage,'slots.jsonl'),guard=budget,policy_hash=cfg.server.thinking.protocol_hash(),special_token_ids=backend.special_token_ids)
        report.extend(r.outcomes)
        times.append({'prompt_id':plan.prompt_id,'seconds':time.monotonic()-t,'executed':r.executed_slots,'cached':r.skipped_slots})
        atomic_write_json(root/stage/f'worker{index}.json',{'outcomes':[dataclasses.asdict(x) for x in report],'timings':times,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'pid':os.getpid(),'gpu':index})
        print(json.dumps({'worker':index,'prompts':len(report),'executed':sum(x['executed'] for x in times),'active_minutes':budget.elapsed()/60}),flush=True)
    del model
    torch.cuda.empty_cache()

def supervise(root,stage,workers=4):
    requests=json.loads((root/stage/'requests.json').read_text())
    planned=sum(len(p['slot_ids']) for p in requests)
    if len(list((root/stage/'slots').glob('*.json')))==planned:
        aggregate(root,stage)
        return
    budgetpath=root/'gpu_budget.json'
    previous=json.loads(budgetpath.read_text())
    prior=previous['active_seconds'];priorhours=previous.get('gpu_hours',prior/3600)
    if prior>=175*60:raise RuntimeError('No new GPU tasks after 175 minutes')
    started=time.monotonic();last=started;hours=priorhours
    children=[];logs=[]
    try:
        for i in range(workers):
            log=(root/stage/f'worker{i}.log').open('a');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
            children.append(subprocess.Popen([sys.executable,'-m','aimo.real_parallel','worker',str(root),str(i),str(workers),stage],env=env,stdout=log,stderr=subprocess.STDOUT))
        while True:
            now=time.monotonic();alive=[p for p in children if p.poll() is None]
            hours+=(now-last)*len(alive)/3600;last=now
            elapsed=prior+now-started
            atomic_write_json(budgetpath,{'active_seconds':elapsed,'gpu_hours':hours,'updated_at':time.time(),'active_workers':len(alive),'owned_pids':[p.pid for p in children],'limit_minutes':180,'behavior_cutoff_minutes':120,'new_gpu_cutoff_minutes':175})
            if not alive:break
            if elapsed>=180*60:
                for p in alive:p.terminate()
                for p in alive:
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                break
            time.sleep(2)
        atomic_write_json(root/stage/'supervisor.json',{'workers':workers,'returncodes':[p.returncode for p in children],'stage_wall_seconds':time.monotonic()-started,'prior_active_seconds':prior,'owned_pids':[p.pid for p in children]})
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for log in logs:log.close()
    aggregate(root,stage)

def aggregate(root,stage):
    import numpy as np
    requests=json.loads((root/stage/'requests.json').read_text())
    raw=[json.loads(p.read_text()) for p in (root/stage/'slots').glob('*.json')]
    counts=collections.Counter(r.get('outcome','interrupted') for r in raw)
    planned=sum(len(p['slot_ids']) for p in requests);counts['not_started']=planned-len(raw)
    results=[r['result'] for r in raw if 'result' in r]
    budget=json.loads((root/'gpu_budget.json').read_text())
    timings=[];peak=0
    for p in (root/stage).glob('worker*.json'):
        d=json.loads(p.read_text());timings+=d['timings'];peak=max(peak,d['peak_vram_bytes'])
    service=sum(t['seconds'] for t in timings);executed=sum(t['executed'] for t in timings)
    q=lambda key:dict(zip(['p50','p90','max'],map(float,np.percentile([r[key] for r in results],[50,90,100])))) if results else {}
    summary={'counts':dict(counts),'planned_slots':planned,'actual_slots':len(raw),'gpu_active_minutes':budget['active_seconds']/60,'gpu_hours':budget['gpu_hours'],'generated_token_quantiles':q('generated_tokens'),'prompt_token_quantiles':q('prompt_tokens'),'single_worker_seconds_per_slot':service/max(1,executed),'new_executed_slots':executed,'worker_service_seconds':service,'peak_vram_bytes':peak,'model':REV,'policy_hash':requests[0]['policy_hash'],'workers':4,'external_stop_slots':sum(r.get('termination')=='external_stop' for r in raw),'explicit_stop_test_slots':1,'cost_gate_interrupted_slots':max(0,sum(r.get('termination')=='external_stop' for r in raw)-1),'unsupported_fraction':sum(r.get('outcome')=='U_score' and r.get('termination')!='external_stop' for r in raw)/max(1,len(raw))}
    atomic_write_json(root/stage/'summary.json',summary)
    record(root,stage.upper()+'_COMPLETED' if len(raw)==planned else stage.upper()+'_PARTIAL',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    mode=sys.argv[1];root=Path(sys.argv[2])
    if mode=='worker':worker(root,int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])
    else:supervise(root,sys.argv[3])
