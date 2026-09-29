"""FP32 observation mapping, frozen-label training and held-out evaluation."""
import sys,json,time,dataclasses,collections
from pathlib import Path
import torch,numpy as np
from .real_run import cfg_for,sha,record,REV
from .runtime import atomic_write_json,DedupLedger
from .bf16_budget import Guard,run
from .bf16_report import summarize

def page_worker(root):
 from .adapters.qwen import load_real_qwen,QwenGenerationBackend,ExtractionRequest,select_landmarks,run_page_extraction
 from .page import load_pages
 torch.set_num_threads(2);policy=json.loads((root/'protocol/policy.json').read_text());obs=policy['observation_provenance'];guard=Guard(root,175)
 index=json.loads((root/'pages/index.json').read_text()) if (root/'pages/index.json').exists() else {};old=Path(policy['parent_run']);reference_model_configs=set()
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(cfg_for(root).server.thinking.model_path,local_files_only=True)
 oldreq={p['prompt_id']:p for p in json.loads((old/'calibration/requests.json').read_text())}
 # Reuse old calibration pages by reference, never copied or relabelled as training.
 for path in (old/'calibration/audit_pages').glob('*.npz'):
  checksum=sha(path.read_bytes());assert path.with_suffix('.sha256').read_text()==checksum
  page=load_pages(path)[0]
  assert all(page.provenance[k]==obs[k] for k in ['source','policy_hash','model_hash','tokenizer_hash','template_hash'])
  assert page.provenance['dtype']=='float32' and page.provenance['backend']=='Qwen3Model' and page.provenance['extractor']=='aimo.adapters.qwen.extract_page' and page.provenance['n_layers']==36
  reference_model_configs.add(page.provenance['config_hash'])
  plan=oldreq[page.variant_id]
  assert page.provenance['prompt_len']==len(plan['rendered_input_ids'])
  text=tok.apply_chat_template([{'role':'user','content':plan['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
  assert tok(text,add_special_tokens=False)['input_ids']==plan['rendered_input_ids']
  spans=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];start=text.index(plan['prompt']);end=start+len(plan['prompt'])
  body=[i for i,(a,b) in enumerate(spans) if a>=start and b<=end and b>a]
  offsets,valid,rel=select_landmarks([body[round(j*(len(body)-1)/15)] for j in range(16)],len(spans))
  assert torch.equal(page.token_offsets,offsets) and torch.equal(page.valid,valid) and torch.equal(page.relative_positions,rel)
  index[page.variant_id]={'path':str(path),'sha256':checksum,'input_token_hash':sha(json.dumps(plan['rendered_input_ids'])),'original_id':page.original_id,'content_fingerprint':page.content_fingerprint(),'split':'calibration','reused':True}
 assert len(reference_model_configs)==1
 expected_model_config_hash=next(iter(reference_model_configs))
 atomic_write_json(root/'pages/index.json',index)
 groups=json.loads((root/'collection/groups.json').read_text())
 expected=[p for g in groups for p in g['prompts']]
 if all(p['prompt_id'] in index and Path(index[p['prompt_id']]['path']).exists() for p in expected):
  for p in expected:
   item=index[p['prompt_id']];path=Path(item['path'])
   assert path.exists() and sha(path.read_bytes())==item['sha256'] and item['input_token_hash']==p['input_token_hash']
  atomic_write_json(root/'pages/resume_cache_check.json',{'all_main_pages_reused':len(expected),'model_loaded':False})
  return
 if guard.should_stop()[0]:return
 t=time.monotonic();cfg=cfg_for(root);atomic_write_json(root/'pages/extraction_config.json',cfg.to_dict());model,tok=load_real_qwen(cfg);backend=QwenGenerationBackend(cfg.server.thinking,model,tok)
 from .page import provenance_hash
 actual_model_config_hash=provenance_hash(model.config.to_dict());assert actual_model_config_hash==expected_model_config_hash
 assert sha(tok.chat_template)==obs['template_hash'] and sha(tok.backend_tokenizer.to_str())==obs['tokenizer_hash']
 groups=json.loads((root/'collection/groups.json').read_text());extracted=0
 for g in groups:
  for p in g['prompts']:
   if guard.should_stop()[0]:break
   rendered=backend.render(p['prompt']);assert rendered['input_ids'][0].tolist()==p['input_ids']
   text=rendered['text'];start=text.index(p['prompt']);end=start+len(p['prompt'])
   spans=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];body=[i for i,(a,b) in enumerate(spans) if a>=start and b<=end and b>a]
   offsets,valid,rel=select_landmarks([body[round(j*(len(body)-1)/15)] for j in range(16)],len(spans))
   req=ExtractionRequest(g['original_id'],p['prompt_id'],rendered['input_ids'].to('cuda'),offsets,valid,rel)
   prov={**obs,'config_hash':actual_model_config_hash,'run_config_hash':cfg.hash(),'reference_run_config_hash':obs['config_hash'],'input_token_hash':p['input_token_hash'],'observation_forward_dtype':'float32','target_model_revision':REV}
   pages,report=run_page_extraction(model,[req],ledger=DedupLedger.open(root/'pages','ledger.jsonl'),guard=guard,provenance=prov,output_dir=root/'pages/artifacts')
   if report['errors'] or len(pages)!=1:raise RuntimeError(report)
   key=sha(json.dumps({'variant':req.variant_id,'tokens':req.input_ids.cpu().tolist(),'provenance':prov,'offsets':offsets.tolist()},sort_keys=True));path=root/'pages/artifacts'/(key+'.npz');page=pages[0]
   index[p['prompt_id']]={'path':str(path),'sha256':sha(path.read_bytes()),'input_token_hash':p['input_token_hash'],'original_id':g['original_id'],'content_fingerprint':page.content_fingerprint(),'split':g['split'],'reused':report['skipped']==1}
   atomic_write_json(root/'pages/index.json',index);extracted+=report['extracted']
 atomic_write_json(root/'pages/summary.json',{'reused_calibration':18,'new_main_pages':extracted,'indexed_pages':len(index),'wall_seconds':time.monotonic()-t,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'forward_dtype':'float32','storage_dtype':'float32','model_config_hash':actual_model_config_hash})
 del model;torch.cuda.empty_cache()

def datasets(root, behavior_dtype='bfloat16'):
 from .page import load_pages
 from .data import OriginalGroup,PageDataset,attach_labels
 from .labels import PairLabel,LabelStore,PanelCoverage,build_panel_label
 policy=json.loads((root/'protocol/policy.json').read_text());index=json.loads((root/'pages/index.json').read_text());groups=json.loads((root/'collection/groups.json').read_text())
 labels=[PairLabel(**l) for l in json.loads((root/'labels/collection_pairs.json').read_text())];store=LabelStore(pairs={l.variant_id:l for l in labels},policy_hash=policy['behavior_policy_id'],scorer_version=policy['profile']['scorer_version'])
 mapping={'behavior_policy_id':policy['behavior_policy_id'],'observation_policy_id':policy['observation_policy_id'],'target_model_revision':REV,'tokenizer_hash':policy['observation_provenance']['tokenizer_hash'],'template_hash':policy['observation_provenance']['template_hash'],'observation_dtype':'float32','behavior_dtype':behavior_dtype,'target_model_config_hash':json.loads((root/'pages/summary.json').read_text())['model_config_hash'],'pages':index}
 splits=collections.defaultdict(list)
 for g in groups:
  if any(p['prompt_id'] not in index for p in g['prompts']):continue
  pages=[]
  for p in g['prompts']:
   item=index[p['prompt_id']];path=Path(item['path']);assert sha(path.read_bytes())==item['sha256'] and item['input_token_hash']==p['input_token_hash']
   page=load_pages(path)[0];assert page.content_fingerprint()==item['content_fingerprint'];pages.append(page)
  labs=[store.pairs[p['prompt_id']] for p in g['prompts'][1:]];members=[p['prompt_id'] for p in g['prompts'][1:]]
  from .bf16_collect import slot_path
  measured=[p['prompt_id'] for p in g['prompts'][1:] if any(slot_path(root,'collection',slot['slot_id']).exists() for slot in p['slots'])]
  coverage=PanelCoverage(expected_members=members,actual_members=[l.variant_id for l in labs if l.has_drop],page_members=members,outcome_members=measured)
  store.panels[g['original_id']]=build_panel_label(g['original_id'],g['original_id'],labs,coverage,policy_hash_value=policy['behavior_policy_id'])
  split='known_test' if g['split']=='known_original_test' else g['split']
  splits[split].append(OriginalGroup(g['original_id'],pages[0],pages[1:],g['original_id'],metadata={'topic':g['topic'],'difficulty':g['difficulty']}))
 ds={s:PageDataset(s,gs) for s,gs in splits.items()};attachment=attach_labels(ds,store,observation_behavior_mapping=mapping)
 store.save(root/'labels/store.json');atomic_write_json(root/'protocol/observation_behavior_mapping.json',mapping);atomic_write_json(root/'labels/attachment.json',attachment)
 return ds

def train_worker(root,method):
 from .train import train
 torch.set_num_threads(2);ds=datasets(root);cfg=cfg_for(root);cfg.run.run_id=method+'_seed0';cfg.paths.output_root=str(root/'training');cfg.data.source='pages';cfg.model.name=method;cfg.train.task='joint' if method=='joint' else 'behavior';cfg.train.w_robust=0;cfg.train.w_pair_drop=1;cfg.train.w_max_drop=0;cfg.train.use_max_drop=False;cfg.train.select_metric='L_pair_drop';cfg.train.microbatch_originals=1;cfg.eval.bootstrap_samples=2000
 atomic_write_json(root/'training'/(method+'_config.json'),cfg.to_dict())
 start=time.monotonic();summary=train(cfg,ds,resume=(cfg.run_dir/'last.pt').exists(),guard=Guard(root,175));summary['wall_seconds']=time.monotonic()-start;summary['peak_vram_bytes']=torch.cuda.max_memory_allocated();atomic_write_json(cfg.run_dir/'summary.json',summary)

def evaluate_worker(root):
 from .train import load_checkpoint
 from .evaluate import evaluate_behavior,evaluate_dataset
 torch.set_num_threads(2);ds=datasets(root);guard=Guard(root,175);outputs={}
 for method in ['constant','raw_change','behavior_m0','behavior','joint']:
  if guard.should_stop()[0]:break
  model,stats,payload=load_checkpoint(root/'training'/(method+'_seed0')/'best.pt');model=model.to('cuda');stats=stats.to('cuda')
  result=evaluate_behavior(model,ds['known_test'],stats,bootstrap_samples=2000,seed=0,support_swap=True,microbatch=1)
  outputs[method]=result;atomic_write_json(root/'evaluation'/(method+'.json'),result)
  del model,stats;torch.cuda.empty_cache()
 paired={};rng=np.random.default_rng(0)
 if 'joint' in outputs:
  j=outputs['joint']['per_original_metrics']['pair_drop_mae']
  for other in ['constant','raw_change','behavior_m0','behavior']:
   if other not in outputs:continue
   b=outputs[other]['per_original_metrics']['pair_drop_mae'];ids=sorted(set(j)&set(b));delta=np.array([b[i]-j[i] for i in ids]);boot=delta[rng.integers(0,len(ids),(2000,len(ids)))].mean(1)
   paired[other]={'joint_mae_improvement':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'n_originals':len(ids),'interpretation':'conditional on observed four-slot labels; not true correctness probability CI'}
 atomic_write_json(root/'evaluation/paired.json',paired)
 # Flow evaluation is separate and cannot influence checkpoint selection.
 if 'joint' in outputs and not guard.should_stop()[0]:
  model,stats,payload=load_checkpoint(root/'training/joint_seed0/best.pt');model=model.to('cuda');stats=stats.to('cuda')
  flow=evaluate_dataset(model,ds['known_test'],stats,horizons=(2,3,4),bootstrap_samples=2000,support_swap=True,seed=0)
  if not guard.should_stop()[0]:
   from .train import validation_metrics
   from .config import config_from_dict
   flow['exact_loss_terms']=validation_metrics(model,ds['known_test'],stats,config_from_dict(payload['config']))
  atomic_write_json(root/'evaluation/joint_flow.json',flow)
 atomic_write_json(root/'evaluation/completed.json',{'methods':list(outputs),'paired':paired,'seed_variance':None,'seed_note':'seed0 only; no test-dependent expansion'})

def _post(root):
 summary=summarize(root,'collection');run(root,[(0,[sys.executable,'-m','aimo.bf16_post','pages',str(root)])],'pages')
 mapped=datasets(root)
 counts={('known_original_test' if split=='known_test' else split):sum(any(l.has_drop for l in g.pair_labels.values()) for g in data.groups) for split,data in mapped.items()}
 atomic_write_json(root/'labels/page_valid_operational_counts.json',{'label_and_page_valid_originals':counts,'raw_label_valid_originals':summary['label_valid_originals']})
 del mapped
 minimum={'train':24,'validation':8,'known_original_test':12};shortfall={s:max(0,n-counts.get(s,0)) for s,n in minimum.items()}
 if any(shortfall.values()):
  record(root,'DATA/PROTOCOL_LIMIT',{'collection':summary,'training':'not run: insufficient label-valid originals','shortfall':shortfall,'architecture':'untested','pages':json.loads((root/'pages/summary.json').read_text()) if (root/'pages/summary.json').exists() else None});atomic_write_json(root/'runtime/completed.json',{'training':False,'shortfall':shortfall})
  from .bf16_finalize import finalize
  finalize(root);return
 commands=[(gpu,[sys.executable,'-m','aimo.bf16_post','train',str(root),methods]) for gpu,methods in [(0,'constant,raw_change'),(1,'behavior_m0'),(2,'behavior'),(3,'joint')]]
 codes=run(root,commands,'training')
 if not codes or any(c!=0 for c in codes):
  record(root,'TRAINING_INTERRUPTED',{'methods':'all seed0 comparisons','codes':codes});return
 run(root,[(0,[sys.executable,'-m','aimo.bf16_post','evaluate',str(root)])],'evaluation')
 record(root,'EVALUATION_RECORDED',{'collection':summary,'evaluation':json.loads((root/'evaluation/completed.json').read_text())});atomic_write_json(root/'runtime/completed.json',{'training':True})
 from .bf16_finalize import finalize
 finalize(root)
def post(root):
 try:
  return _post(root)
 except Exception as exc:
  import traceback
  atomic_write_json(root/'runtime/post_error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
  raise
 finally:
  from .bf16_finalize import finalize
  finalize(root)

if __name__=='__main__':
 mode=sys.argv[1];root=Path(sys.argv[2])
 if mode=='pages':page_worker(root)
 elif mode=='train':
  for method in sys.argv[3].split(','):
   if Guard(root,175).should_stop()[0]:break
   free,total=torch.cuda.mem_get_info()
   if free<16*1024**3:raise RuntimeError('Predictor GPU has insufficient safe VRAM')
   train_worker(root,method);torch.cuda.empty_cache()
 elif mode=='evaluate':evaluate_worker(root)
 elif mode=='post':post(root)
