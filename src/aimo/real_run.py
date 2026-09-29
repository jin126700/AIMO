"""Frozen DeepMath real-data orchestration; all artifacts stay in the experiment root."""
from pathlib import Path
import collections
import dataclasses
import hashlib
import json
import os
import time
import fcntl
import torch
from .runtime import atomic_write_json, atomic_write_text, DedupLedger
from .config import Config

REV='1cfa9a7208912126459214e8b04321603b3df60c'
DATA=Path('/data1/Data/AIMO/Datasets/prepared/aimo_v2/20260927_01')
REPO=Path('/data1/HKM/AIMO')
def sha(x):
    return hashlib.sha256(x if isinstance(x,bytes) else str(x).encode()).hexdigest()
def read_rows(p):
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]
def cfg_for(root):
    cfg=Config();cfg.run.device='cuda';cfg.run.run_id=root.name
    cfg.paths.output_root=str(root.parent)
    p=cfg.server.thinking
    p.model_revision=REV;p.model_path=f'/data1/Data/AIMO/Models/Qwen/Qwen3-4B/{REV}'
    p.max_new_tokens=16384;p.samples_per_prompt=2;p.max_total_context=40960
    p.numerical_backend='fp32_sdpa';p.scorer_id='exact_integer_rational';p.scorer_version='2'
    p.final_cap_failure_policy='censored_X'
    return cfg

def record(root,status,details):
    atomic_write_json(root/'status.json',{'status':status,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**details})
    text=f'## {root.name}\n\n- 목적: DeepMath pair-drop prediction prototype.\n- 상태: {status}\n- 원격 SHA: 80f257ded148b3e199769aec6b6e4654d88915c6; local patch hash는 protocol/patch.sha256 참조.\n- 데이터/model/policy: protocol/manifest.json 및 calibration/config.json 참조.\n- 결과 경로: {root}\n\n```json\n{json.dumps(details,ensure_ascii=False,indent=2)}\n```\n'
    atomic_write_text(root/'report.md',text)
    begin=f'<!-- run:{root.name}:start -->';end=f'<!-- run:{root.name}:end -->'
    path=REPO/'Loop_result.md'
    with (REPO/'.Loop_result.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=path.read_text() if path.exists() else '# AIMO 실험 기록\n'
        section=begin+'\n'+text+end
        if begin in old:
            a=old.index(begin);b=old.index(end,a)+len(end);old=old[:a]+section+old[b:]
        else:old+='\n'+section+'\n'
        atomic_write_text(path,old)

def prepare(root):
    from transformers import AutoTokenizer
    from .scoring import compare_answers
    tok=AutoTokenizer.from_pretrained(cfg_for(root).server.thinking.model_path,local_files_only=True)
    inputs={r['input_id']:r['question'] for r in read_rows(DATA/'inputs.jsonl')}
    answers={r['input_id']:r for r in read_rows(DATA/'answers.jsonl')}
    candidates=read_rows(DATA/'candidates.jsonl')
    frozen=json.loads((DATA/'splits.json').read_text())['splits']
    pairs=read_rows(DATA/'pairs.jsonl');by=collections.defaultdict(list)
    for p in pairs:by[p['original_id']].append(p)
    groups=collections.defaultdict(list);audit=collections.Counter();seen=set()
    for c in candidates:
        oid=c['original_id'];split=c['split']
        if split not in ('calibration','train','validation','known_original_test'):continue
        assert oid in frozen[split]
        assert oid not in seen;seen.add(oid)
        q=inputs[oid];gold=answers[oid]['answer_raw']
        if compare_answers(gold,gold)!='correct':audit['unsupported_answer']+=1;continue
        origids=tok.apply_chat_template([{'role':'user','content':q}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)
        variants=[];token_seen={tuple(origids)}
        for p in sorted(by[oid],key=lambda p:sha(p['variant_id'])):
            if p['semantic_status'] not in ('verified_by_construction','verified_by_existing_evidence'):continue
            v=inputs[p['variant_id']];ev=p['evidence']
            assert sha(q)==ev['original_sha256'] and sha(v)==ev['variant_sha256']
            assert q.split()==v.split() and ev['math_spans_byte_identical']
            ids=tok.apply_chat_template([{'role':'user','content':v}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)
            if tuple(ids) in token_seen:audit['token_identity']+=1;continue
            token_seen.add(tuple(ids));variants.append({'id':p['variant_id'],'question':v,'tokens':ids,'pair':p})
        if len(variants)<2:audit[split+'_shortfall']+=1;continue
        groups[split].append({'id':oid,'split':split,'topic':c['field'],'difficulty':c['difficulty'],'question':q,'gold':gold,'tokens':origids,'variants':variants[:2]})
    # Topic-interleaved deterministic hash order; outcomes never consulted.
    ordered={}
    for split,gs in groups.items():
        buckets=collections.defaultdict(list)
        for g in sorted(gs,key=lambda g:sha(g['id'])):buckets[g['topic']].append(g)
        seq=[]
        while any(buckets.values()):
            for topic in sorted(buckets):
                if buckets[topic]:seq.append(buckets[topic].pop(0))
        ordered[split]=seq
    small={'train':32,'validation':8,'known_original_test':12}
    standard={'train':40,'validation':16,'known_original_test':24}
    cohorts={name:{s:[g['id'] for g in ordered.get(s,[])[:n]] for s,n in sizes.items()} for name,sizes in [('small',small),('standard',standard)]}
    out={'calibration':ordered.get('calibration',[])[:6],'ordered':ordered,'cohorts':cohorts,'available':{s:len(v) for s,v in ordered.items()},'audit':dict(audit),'prepared_hashes':{n:sha((DATA/n).read_bytes()) for n in ['inputs.jsonl','answers.jsonl','metadata.parquet','pairs.jsonl','splits.json','manifest.json']},'template_hash':sha(tok.chat_template),'model_revision':REV,'selection':'hash_then_topic_round_robin_before_outcomes','test_use':'known_original_test only; unseen/harder excluded'}
    print(json.dumps({'available':out['available'],'audit':out['audit']}),flush=True)
    atomic_write_json(root/'protocol/preparation_audit.json',out)
    assert len(out['calibration'])==6
    assert all(len(cohorts['small'][s])==n for s,n in small.items()),out['available']
    p=root/'protocol/frozen.json'
    if p.exists():assert json.loads(p.read_text())==out,'Frozen protocol mismatch'
    else:atomic_write_json(p,out)
    atomic_write_json(root/'calibration/config.json',cfg_for(root).to_dict())
    record(root,'PREPARED',{'available_two_distinct_variants':out['available'],'audit':out['audit'],'calibration_planned':36,'gpu_active_minutes':0,'label_valid_originals':0,'label_valid_pairs':0})
    print(json.dumps({'available':out['available'],'audit':out['audit']}),flush=True)

class SharedBudget:
    def __init__(self,root,limit=180):
        self.root=root;self.path=root/'gpu_budget.json';self.limit=limit
        data=json.loads(self.path.read_text()) if self.path.exists() else {}
        self.prior=data.get('active_seconds',0);self.start=time.monotonic()
        self.save()
    def elapsed(self):return self.prior+time.monotonic()-self.start
    def save(self):atomic_write_json(self.path,{'active_seconds':self.elapsed(),'gpu_hours':self.elapsed()/3600,'gpu_count':1,'limit_minutes':180,'behavior_cutoff_minutes':120,'new_gpu_cutoff_minutes':175,'pid':os.getpid()})
    def should_stop(self):
        e=self.elapsed();stop=(self.root/'STOP').exists() or e>=self.limit*60
        if stop:self.save()
        return stop,'budget_or_external_stop' if stop else ''

def calibration(root):
    from .adapters.qwen import load_real_qwen,QwenGenerationBackend,ExtractionRequest,run_page_extraction,select_landmarks
    from .collect import CollectionPlan,collect_outcomes
    frozen=json.loads((root/'protocol/frozen.json').read_text());cfg=cfg_for(root)
    budget=SharedBudget(root,120)
    if budget.elapsed()>=175*60:raise RuntimeError('No new GPU work after 175 minutes')
    try:
        torch.set_num_threads(4)
        model,tok=load_real_qwen(cfg);backend=QwenGenerationBackend(cfg.server.thinking,model,tok)
        backend.stop_check=lambda:budget.should_stop()[0]
        policy=cfg.server.thinking.protocol_hash();prov={'source':'qwen3_real','policy_hash':policy,'model_hash':REV,'tokenizer_hash':sha(tok.backend_tokenizer.to_str()),'template_hash':sha(tok.chat_template),'config_hash':cfg.hash()}
        # Actual prompt-side landmark audit before expensive generation.
        g=frozen['calibration'][0];rendered=backend.render(g['question']);text=rendered['text']
        start=text.index(g['question']);finish=start+len(g['question'])
        offsets=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
        body=[i for i,(a,b) in enumerate(offsets) if a>=start and b<=finish and b>a]
        positions=[body[round(j*(len(body)-1)/15)] for j in range(16)]
        landmarks,valid,rel=select_landmarks(positions,len(offsets))
        req=ExtractionRequest(g['id'],g['id'],rendered['input_ids'].to('cuda'),landmarks,valid,rel)
        pages,report=run_page_extraction(model,[req],provenance=prov,output_dir=root/'calibration/audit_pages')
        if report['errors'] or not pages:raise RuntimeError(report)
        from .adapters.qwen import extract_page
        again=extract_page(model,req.input_ids,landmarks,valid,rel,g['id'],g['id'],provenance=prov)
        noise=float((pages[0].state-again.state).abs().max())
        audit={'shape':list(pages[0].state.shape),'residual_error':pages[0].residual_identity_error(),'repeat_max_abs':noise,'body_landmark_offsets':positions,'all_body_landmarks_in_question':all(i in body for i in positions),'final_prompt_token':tok.convert_ids_to_tokens(rendered['input_ids'][0,-1].item()),'provenance':prov,'model_device':str(next(model.parameters()).device),'dtype':str(next(model.parameters()).dtype),'gpu_name':torch.cuda.get_device_name(),'vram_peak_bytes':torch.cuda.max_memory_allocated()}
        atomic_write_json(root/'calibration/numerical_audit.json',audit)
        plans=[]
        for g in frozen['calibration']:
            for p in [{'id':g['id'],'question':g['question'],'tokens':g['tokens']}]+g['variants']:
                assert backend.render(p['question'])['input_ids'][0].tolist()==p['tokens']
                plans.append(CollectionPlan(p['id'],p['question'],g['gold'],[f'calibration:{p["id"]}:{i}' for i in range(2)],[int(sha(f'calibration:{p["id"]}:{i}')[:8],16) for i in range(2)]))
        atomic_write_json(root/'calibration/requests.json',[{**dataclasses.asdict(p),'policy_hash':policy,'model_revision':REV,'rendered_input_ids':backend.render(p.prompt)['input_ids'][0].tolist()} for p in plans])
        start=time.monotonic();reports=[]
        for p in plans:
            r=collect_outcomes(backend,[p],cfg.server.thinking,ledger=DedupLedger.open(root/'calibration','slots.jsonl'),guard=budget,policy_hash=policy,special_token_ids=backend.special_token_ids)
            reports.extend(r.outcomes);budget.save()
            atomic_write_json(root/'calibration/outcomes.json',[dataclasses.asdict(x) for x in reports])
            print(json.dumps({'completed_prompts':len(reports),'counts':dict(sum((collections.Counter(x.counts) for x in reports),collections.Counter())),'elapsed_min':budget.elapsed()/60}),flush=True)
        counts=dict(sum((collections.Counter(x.counts) for x in reports),collections.Counter()))
        raw=[json.loads(p.read_text()) for p in (root/'calibration/slots').glob('*.json')]
        lens=[r['result']['generated_tokens'] for r in raw if 'result' in r]
        import numpy as np
        elapsed=time.monotonic()-start
        summary={'counts':counts,'planned_slots':36,'actual_slots':len(raw),'wall_seconds':elapsed,'gpu_active_minutes':budget.elapsed()/60,'generated_tokens_quantiles':dict(zip(['p50','p90','max'],map(float,np.percentile(lens,[50,90,100])))) if lens else {},'single_worker_seconds_per_slot':elapsed/max(1,len(raw)),'peak_vram_bytes':torch.cuda.max_memory_allocated(),'model':REV,'policy_hash':policy,'audit':audit}
        atomic_write_json(root/'calibration/summary.json',summary)
        record(root,'CALIBRATION_COMPLETED' if len(raw)==36 else 'CALIBRATION_PARTIAL',summary)
    finally:budget.save()

def command(args):
    root=Path(args.run_dir)
    if args.stage=='prepare':prepare(root)
    elif args.stage=='calibration':
        if not args.execute_gpu:raise ValueError('--execute-gpu required')
        if (root/"calibration/numerical_audit.json").exists():
            from .real_parallel import supervise
            supervise(root,"calibration")
        else:
            calibration(root)
    elif args.stage=="audit-resume":
        from .real_report import audit_resume
        print(json.dumps(audit_resume(root),ensure_ascii=False))
    else:finish(root)
    return 0


def finish(root):
    """Artifact-based final accounting. Does not invent labels or trigger new cohorts."""
    cal=root/'calibration/summary.json'
    raw=[json.loads(p.read_text()) for p in (root/'calibration/slots').glob('*.json')]
    counts=dict(collections.Counter(r.get('outcome','interrupted') for r in raw))
    budget=json.loads((root/'gpu_budget.json').read_text()) if (root/'gpu_budget.json').exists() else {}
    details={'label_valid_originals':0,'label_valid_pairs':0,'main_cohort_started':False,
             'calibration_counts':counts,'calibration_started_slots':len(raw),
             'calibration_planned_slots':36,'main_planned_slots':{'small':624,'standard':960},
             'gpu_active_minutes':budget.get('active_seconds',0)/60,
             'models_evaluated':[],'joint_vs_behavior_only':None,'joint_vs_m0':None,
             'pair_specific_signal':None,'robust_head':'null/untrained',
             'training_seed_variance':None,'judgment':'DATA/PROTOCOL_LIMIT'}
    if cal.exists():
        summary=json.loads(cal.read_text());seconds=summary['single_worker_seconds_per_slot']
        # Optimistic ideal four-worker bound is reported, not assumed measured throughput.
        elapsed=budget.get('active_seconds',0)
        details['runtime_estimate_minutes']={str(n):{'one_worker':n*seconds/60,
              'ideal_four_workers':n*seconds/240} for n in (528,624,960)}
        details['available_behavior_minutes']=max(0,120-elapsed/60)
        details['calibration_summary']=str(cal)
        if elapsed+528*seconds/4 < 120*60:
            raise RuntimeError('Cost may fit: continue collection/training; cannot finalize as budget blocker')
        details['blocker']='Even ideal four-worker extrapolation for the 528-slot operational minimum exceeds the remaining behavior budget; no main cohort launched.'
    else:
        details['blocker']='Calibration incomplete; detailed execution error is retained in calibration/run.log.'
    from .real_report import calibration_labels
    details['calibration_label_summary']=calibration_labels(root)
    details['comparison_table']={m:{'MAE':None,'Huber':None,'status':'not_trained_insufficient_budget_for_labels'} for m in ['constant','raw_change','behavior_m0','behavior','joint']}
    details['cost_gate']=json.loads((root/'protocol/cost_gate.json').read_text())
    details['calibration_page_report']=json.loads((root/'calibration/page_report.json').read_text())
    details['gpu_hours']=budget.get('gpu_hours')
    details['main_unstarted_ratio']=1.0
    details['coverage_limitation']='No main labels were generated; calibration-only performance cannot establish pair-specific behavior signal.'
    atomic_write_json(root/'evaluation/result.json',details)
    record(root,'DATA/PROTOCOL_LIMIT',details)
    return details
