import json,collections,dataclasses,time
from pathlib import Path
import numpy as np
from .runtime import atomic_write_json
from .bf16_collect import slot_path,identity
from .real_run import record
from .labels import PromptOutcome,pair_drop,SEMANTIC_VERIFIED

def summarize(root,stage):
 frozen=json.loads((root/'protocol/frozen.json').read_text());recipes={v['id']:v['pair']['recipe'] for gs in frozen['ordered'].values() for g in gs for v in g['variants']}
 groups=json.loads((root/stage/'groups.json').read_text());policy=json.loads((root/'protocol/policy.json').read_text());counts=collections.Counter();terminations=collections.Counter();raws=[];outcomes={};labels=[];valid_originals=collections.Counter();valid_pairs=collections.Counter();complete_panels=collections.Counter();coverage=collections.defaultdict(collections.Counter)
 for g in groups:
  group_labels=[]
  for p in g['prompts']:
   pc=collections.Counter();evidence={};completed=0
   for s in p['slots']:
    path=slot_path(root,stage,s['slot_id'])
    if not path.exists():pc['not_started']+=1;continue
    raw=json.loads(path.read_text());assert raw['identity']==identity(p,s,policy)
    if raw['state']=='started' and json.loads((root/'gpu_budget.json').read_text()).get('active_workers',0)==0:
     raw.update(state='interrupted',outcome='U_score',termination='interrupted');atomic_write_json(path,raw)
    code=raw.get('outcome','U_score');pc[code]+=1;raws.append(raw);terminations[raw.get('termination','interrupted')]+=1
    completed+=raw['state']=='completed';evidence[s['slot_id']]=raw['identity']
   outcome=PromptOutcome(p['prompt_id'],dict(pc),len(p['slots']),completed,policy_hash=policy['behavior_policy_id'],scorer_version=policy['profile']['scorer_version'],slot_evidence=evidence)
   outcomes[p['prompt_id']]=outcome;counts.update(pc)
   for k in ['split:'+g['split'],'topic:'+g['topic'],'difficulty:'+str(g['difficulty']),'perturbation:'+recipes.get(p['prompt_id'],'original')]:coverage[k].update(pc)
  for p in g['prompts'][1:]:
   lab=pair_drop(outcomes[g['prompts'][0]['prompt_id']],outcomes[p['prompt_id']],original_id=g['original_id'],variant_id=p['prompt_id'],panel_id=g['original_id'],semantic_valid=SEMANTIC_VERIFIED)
   labels.append(dataclasses.asdict(lab));group_labels.append(lab)
  n=sum(l.signed_drop is not None for l in group_labels)
  valid_pairs[g['split']]+=n
  if n:valid_originals[g['split']]+=1
  if n==2:complete_panels[g['split']]+=1
 results=[r['result'] for r in raws if 'result' in r];natural=[r['result'] for r in raws if r.get('termination') in ['eos','token_cap']]
 def q(rs,key):return dict(zip(['p50','p90','max'],map(float,np.percentile([r[key] for r in rs],[50,90,100])))) if rs else {}
 seconds=json.loads((root/stage/'exit.json').read_text()).get('stage_seconds',0) if (root/stage/'exit.json').exists() else 0
 planned=sum(len(p['slots']) for g in groups for p in g['prompts']);resolved=counts['C']+counts['W']
 summary={'stage':stage,'planned_originals':len(groups),'planned_pairs':len(groups)*2,'planned_prompts':len(groups)*3,'planned_slots':planned,'started_slots':len(raws),'started_originals':len({r.get('original_id') for r in raws}),'started_prompts':len({r.get('prompt_id') for r in raws}),'counts':dict(counts),'terminations':dict(terminations),'label_valid_originals':dict(valid_originals),'label_valid_pairs':dict(valid_pairs),'complete_label_panels':dict(complete_panels),'unresolved_fraction':1-resolved/planned,'natural_generated_token_quantiles':q(natural,'generated_tokens'),'prompt_token_quantiles':q(results,'prompt_tokens'),'generated_tokens':sum(x['generated_tokens'] for x in results),'stage_wall_seconds':seconds,'stage_inclusive_tokens_per_second':sum(x['generated_tokens'] for x in results)/seconds if seconds else None,'scored_requests_per_minute':resolved/(seconds/60) if seconds else None,'coverage_by_split_topic_difficulty':{k:dict(v) for k,v in coverage.items()},'budget':json.loads((root/'gpu_budget.json').read_text()),'pair_drop_distribution':dict(collections.Counter(str(l['signed_drop']) for l in labels if l['signed_drop'] is not None)),'wall_time_scope':'shared calibration+main worker stage' if (root/'protocol/transition.json').exists() else stage,'bounds_note':'censoring bounds only; not sampling confidence intervals','robust_head':'null/untrained'}
 timed=[x for x in raws if 'completed_utc' in x and 'batch_elapsed_seconds' in x]
 if timed:
  span=max(x['completed_utc'] for x in timed)-min(x['completed_utc']-x['batch_elapsed_seconds'] for x in timed)
  summary['generation_window_seconds']=span
  summary['generation_window_tokens_per_second']=summary['generated_tokens']/span if span else None
 atomic_write_json(root/stage/'summary.json',summary);atomic_write_json(root/stage/'outcomes.json',[dataclasses.asdict(o) for o in outcomes.values()]);atomic_write_json(root/'labels'/f'{stage}_pairs.json',labels)
 record(root,stage.upper()+'_RECORDED',summary)
 return summary
