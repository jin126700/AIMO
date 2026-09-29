"""CPU-only immutable-evidence summaries; calibration never enters training."""
import collections,json,math
from pathlib import Path
from .adapters.qwen import split_thinking
from .scoring import extract_boxed
from .runtime import atomic_write_json

def calibration_labels(root):
    root=Path(root);plans=json.loads((root/'calibration/requests.json').read_text())
    frozen=json.loads((root/'protocol/frozen.json').read_text())
    raw={}
    for p in (root/'calibration/slots').glob('*.json'):
        row=json.loads(p.read_text());raw[row['identity']['request_id']]=(row,str(p))
    prompts={};evidence=[]
    for p in plans:
        counts=collections.Counter()
        for sid in p['slot_ids']:
            row,path=raw.get(sid,({},None));counts[row.get('outcome','not_started')]+=1
            result=row.get('result',{})
            parsed=split_thinking(result.get('text'),thinking_already_open=result.get('thinking_already_open',False))
            evidence.append({'slot_id':sid,'raw_path':path,'outcome':row.get('outcome','not_started'),'termination':row.get('termination'),'thinking_state':parsed.state,'answer_region':parsed.answer_region,'boxed_candidates':extract_boxed(parsed.answer_region),'gold':p['gold']})
        n=len(p['slot_ids']);c=counts['C'];resolved=counts['C']+counts['W'];u=n-resolved
        interval=None
        if u==0:
            z=1.959963984540054;ph=c/n;den=1+z*z/n
            center=(ph+z*z/(2*n))/den;half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
            interval=[max(0,center-half),min(1,center+half)]
        prompts[p['prompt_id']]={'counts':dict(counts),'planned':n,'resolved':resolved,'coverage':resolved/n,'p_hat':c/n if u==0 else None,'censoring_bounds':[c/n,(c+u)/n],'wilson_sampling_interval_if_uncensored':interval}
    pairs=[];complete=0;by_topic=collections.defaultdict(lambda:collections.Counter())
    for g in frozen['calibration']:
        o=prompts[g['id']];valid=0
        for v in g['variants']:
            w=prompts[v['id']];point=None if o['p_hat'] is None or w['p_hat'] is None else o['p_hat']-w['p_hat']
            valid+=point is not None
            pairs.append({'original_id':g['id'],'variant_id':v['id'],'signed_drop':point,'censoring_bounds':[o['censoring_bounds'][0]-w['censoring_bounds'][1],o['censoring_bounds'][1]-w['censoring_bounds'][0]],'robust_label':None,'calibration_only':True})
        complete+=valid==2
        for pid in [g['id']]+[v['id'] for v in g['variants']]:
            by_topic[g['topic']]['planned_slots']+=prompts[pid]['planned']
            by_topic[g['topic']]['resolved_slots']+=prompts[pid]['resolved']
    summary={'calibration_originals':6,'calibration_pairs':12,'calibration_complete_label_originals':complete,'calibration_label_valid_pairs':sum(p['signed_drop'] is not None for p in pairs),'resolved_slots':sum(p['resolved'] for p in prompts.values()),'planned_slots':36,'unresolved_ratio':1-sum(p['resolved'] for p in prompts.values())/36,'by_topic':dict(by_topic),'main_label_valid_originals':0,'main_label_valid_pairs':0,'note':'Censoring bounds and finite-sample Wilson intervals are different; calibration is excluded from all model training/evaluation.'}
    atomic_write_json(root/'calibration/labels.json',{'summary':summary,'prompts':prompts,'pairs':pairs})
    atomic_write_json(root/'calibration/scorer_evidence.json',evidence)
    return summary


def audit_resume(root):
    from .real_run import sha,REV
    from .collect import prompt_hash
    root=Path(root);plans=json.loads((root/'calibration/requests.json').read_text())
    expected={sid:(p,seed) for p in plans for sid,seed in zip(p['slot_ids'],p['seeds'])}
    raw=[json.loads(p.read_text()) for p in (root/'calibration/slots').glob('*.json')]
    seen=set()
    for row in raw:
        identity=row['identity'];sid=identity['request_id'];assert sid not in seen;seen.add(sid)
        p,seed=expected[sid]
        assert identity['seed']==seed and identity['prompt_hash']==prompt_hash(p['prompt'])
        assert identity['policy_hash']==p['policy_hash'] and identity['model_hash']==REV
        assert identity['scorer_version']=='2' and identity['gold_hash']==prompt_hash(p['gold'])
        assert identity['input_token_hash']==sha(json.dumps(p['rendered_input_ids']))
        if row.get('result',{}).get('input_token_ids') is not None:
            assert row['result']['input_token_ids']==p['rendered_input_ids']
    pages=list((root/'calibration/audit_pages').glob('*.npz'))
    for page in pages:assert sha(page.read_bytes())==page.with_suffix('.sha256').read_text()
    out={'verified_terminal_slots':len(seen),'never_resample_terminal_slots':True,'not_started_slots':len(expected)-len(seen),'verified_pages':len(pages),'gpu_execution_allowed':False,'blocker':'persisted runtime cost gate; STOP retained','gpu_budget':json.loads((root/'gpu_budget.json').read_text())}
    atomic_write_json(root/'protocol/resume_audit.json',out)
    return out
