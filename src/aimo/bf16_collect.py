"""Durable group-level collection, retaining every completed and censored slot."""
import dataclasses,json,hashlib,time,sys,collections
from pathlib import Path
import torch
from .real_run import cfg_for,sha,REV,record
from .runtime import atomic_write_json,DedupLedger,store_lock
from .adapters.qwen import load_real_qwen,SlotRequest,classify_thinking_slot,SCORER_VERSION
from .bf16_batch import BatchedQwen
from .bf16_budget import Guard

def config(root):
 cfg=cfg_for(root);cfg.server.thinking.numerical_backend='bf16_sdpa';cfg.server.thinking.samples_per_prompt=4
 return cfg

def prepare(root,homogeneous=False):
 f=json.loads((root/'protocol/frozen.json').read_text());cfg=config(root)
 import transformers
 policy={'profile':dataclasses.asdict(cfg.server.thinking),'backend':BatchedQwen.version,'transformers':transformers.__version__,'torch':torch.__version__,'rng':'independent_cuda_generator_per_slot','scheduler':'homogeneous_prompt_groups' if homogeneous else 'static_original_group_batch_left_padding','max_batch':4 if homogeneous else 12,'replicas':4,'attention':'sdpa','use_cache':True,'output_hidden_states':False,'output_attentions':False,'output_scores':False}
 pid=sha(json.dumps(policy,sort_keys=True));old=root.parent/'deepmath_real_20260926T193418Z'
 obs=json.loads((old/'calibration/numerical_audit.json').read_text())['provenance']
 policy.update(behavior_policy_id=pid,observation_policy_id=obs['policy_hash'],target_model_revision=REV,observation_provenance=obs,parent_run=str(old),interpretation='FP32 reference Page predicts BF16 behavior; observational, not identical numerical execution',prior_interpretation='행동 측정 비용에 의한 중단; architecture와 robustness 가설 미검증.')
 atomic_write_json(root/'protocol/policy.json',policy)
 # Frozen small cohort fully available; standard requires 24 test but has only 22.
 selected={s:[g for g in f['ordered'][s] if g['id'] in ids] for s,ids in f['cohorts']['small'].items()}
 order=[]
 # Proportional interleave; preserves hash/topic order within each frozen split.
 while any(selected.values()):
  eligible=[s for s in selected if selected[s]]
  s=min(eligible,key=lambda s:(sum(x['split']==s for x in order)/len(f['cohorts']['small'][s]),s))
  order.append(selected[s].pop(0))
 for stage,groups,slots in [('calibration',f['calibration'],2),('collection',order,4)]:
  plans=[]
  for gi,g in enumerate(groups):
   prompts=[]
   for p in [g]+g['variants']:
    ps=[]
    for i in range(slots):
     sid=f'{root.name}:{pid}:{stage}:{p["id"]}:{i}';seed=int(sha(sid)[:8],16)
     ps.append({'slot_id':sid,'seed':seed})
    prompts.append({'prompt_id':p['id'],'prompt':p['question'],'gold':g['gold'],'input_ids':p['tokens'],'input_token_hash':sha(json.dumps(p['tokens'])),'slots':ps})
   plans.append({'original_id':g['id'],'split':g['split'],'topic':g['topic'],'difficulty':g['difficulty'],'order':gi,'prompts':prompts})
  atomic_write_json(root/stage/'groups.json',plans)
 atomic_write_json(root/'protocol/selection.json',{'cohort':'small','reason':'standard frozen known-test has 22/24 eligible originals; small 32/8/12 available; no outcome selection','counts':{s:len(ids) for s,ids in f['cohorts']['small'].items()},'additional_seeds':'seed0 first; only remaining budget can permit 1/2','minimum_label_originals':{'train':24,'validation':8,'known_original_test':12}})
 record(root,'BF16_PREPARED',{'parent':str(old),'inherited_gpu_minutes':json.loads((root/'gpu_budget.json').read_text())['active_seconds']/60,'behavior_policy_id':pid,'observation_policy_id':obs['policy_hash'],'main_planned_originals':52,'main_planned_pairs':104,'main_planned_slots':624,'calibration_slots':36})

def slot_path(root,stage,sid):return root/stage/'slots'/(sha(sid)+'.json')
def identity(p,s,policy):
 return {'request_id':s['slot_id'],'seed':s['seed'],'prompt_hash':sha(p['prompt']),'input_token_hash':p['input_token_hash'],'policy_hash':policy['behavior_policy_id'],'model_hash':REV,'tokenizer_hash':policy['observation_provenance']['tokenizer_hash'],'template_hash':policy['observation_provenance']['template_hash'],'scorer_version':SCORER_VERSION,'gold_hash':sha(p['gold'])}

def collect_group(root,stage,g,backend,policy,guard):
 ledger=DedupLedger.open(root/stage,'slots.jsonl');pending=[];started=time.monotonic()
 with store_lock(root/stage/(sha(g['original_id'])+'.group')):
  for p in g['prompts']:
   if hasattr(backend,'render'):assert backend.render(p['prompt'])['input_ids'][0].tolist()==p['input_ids']
   for s in p['slots']:
    path=slot_path(root,stage,s['slot_id']);ident=identity(p,s,policy)
    if path.exists():
     raw=json.loads(path.read_text());assert raw['identity']==ident,'cached identity mismatch'
     if raw['state']=='started':raw.update(state='interrupted',outcome='U_score',termination='interrupted');atomic_write_json(path,raw)
     continue
    if ledger.seen(s['slot_id']):raise ValueError('ledger without raw evidence')
    pending.append((p,s,path,ident))
  if not pending or guard.should_stop()[0]:return
  for p,s,path,ident in pending:atomic_write_json(path,{'state':'started','identity':ident,'original_id':g['original_id'],'prompt_id':p['prompt_id'],'started_utc':time.time()})
  lookup={s['slot_id']:(p,s,path,ident) for p,s,path,ident in pending}
  def save(result):
   p,s,path,ident=lookup[result.slot_id]
   outcome=classify_thinking_slot(started=result.started,infra_error=result.infra_error,hit_cap=result.hit_cap,is_final_cap=False,text=result.text,gold=p['gold'],thinking_already_open=result.thinking_already_open,generated_token_ids=result.generated_token_ids,special_token_ids=backend.special_token_ids)
   if result.termination_reason=='external_stop':outcome='U_score'
   raw={'identity':ident,'state':'completed','original_id':g['original_id'],'prompt_id':p['prompt_id'],'outcome':outcome,'termination':result.termination_reason,'result':dataclasses.asdict(result),'scorer':{'gold':p['gold'],'version':SCORER_VERSION,'outcome':outcome},'batch_elapsed_seconds':time.monotonic()-started,'completed_utc':time.time()}
   atomic_write_json(path,raw);ledger.mark(result.slot_id,{'artifact':str(path),'outcome':outcome,'identity':ident})
  # Stop during generation is checked at every decoding step. New groups cease at120.
  generation_guard=Guard(root,150)
  backend.stop_check=lambda:generation_guard.should_stop()[0]
  backend.generate_batch([SlotRequest(p['prompt_id'],s['slot_id'],p['prompt'],'',s['seed']) for p,s,path,ident in pending],on_result=save)
  atomic_write_json(root/stage/(sha(g.get('batch_id',g['original_id']))+'.done.json'),{'original_id':g['original_id'],'batch_size':len(pending),'wall_seconds':time.monotonic()-started,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})

def worker(root,stage,index):
 torch.set_num_threads(2);guard=Guard(root,120)
 groups=json.loads((root/stage/'groups.json').read_text())[index::4];policy=json.loads((root/'protocol/policy.json').read_text())
 if guard.should_stop()[0]:return
 free,total=torch.cuda.mem_get_info()
 # 8.2GB weights + BF16 KV (36 layers, 8 heads, head_dim128) + 4GiB headroom.
 batch=6 if stage=='calibration' else 12
 required=8_200_000_000+batch*(16384+199)*36*2*8*128*2+4*1024**3
 if free<required:raise RuntimeError(f'Insufficient safe VRAM for frozen batch: free={free}, required={required}')
 # Resource reservation ceiling only; sampling/model arithmetic is unchanged.
 torch.cuda.set_per_process_memory_fraction(0.90)
 t=time.monotonic();model,tok=load_real_qwen(config(root));backend=BatchedQwen(config(root).server.thinking,model,tok)
 assert sha(tok.chat_template)==policy['observation_provenance']['template_hash'] and sha(tok.backend_tokenizer.to_str())==policy['observation_provenance']['tokenizer_hash']
 atomic_write_json(root/stage/f'load{index}.json',{'cold_seconds':time.monotonic()-t,'dtype':str(next(model.parameters()).dtype),'gpu':torch.cuda.get_device_name(),'replica':index,'hooks':sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules()),'allocator_memory_fraction':0.90,'free_vram_before_load':free,'target_context_required_bytes_including_headroom':required})
 for g in groups:
  collect_group(root,stage,g,backend,policy,guard)
  print(json.dumps({'stage':stage,'replica':index,'original':g['original_id'],'active_minutes':guard.elapsed()/60}),flush=True)
  torch.cuda.empty_cache()
  if guard.should_stop()[0]:break
 del model;torch.cuda.empty_cache()
if __name__=='__main__':
 root=Path(sys.argv[2])
 if sys.argv[1]=='prepare':prepare(root)
 else:worker(root,sys.argv[3],int(sys.argv[4]))
