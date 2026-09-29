"""Second/final BF16 candidate: homogeneous-prompt batches in one frozen queue."""
import sys,json,time,os
from pathlib import Path
import torch
from .bf16_collect import config,collect_group
from .bf16_budget import Guard,run
from .adapters.qwen import load_real_qwen
from .bf16_batch import BatchedQwen
from .runtime import atomic_write_json

def worker(root,index):
 torch.set_num_threads(2);guard=Guard(root,120);policy=json.loads((root/'protocol/policy.json').read_text())
 queue=[(stage,g) for stage in ['calibration','collection'] for g in json.loads((root/stage/'groups.json').read_text())]
 if guard.should_stop()[0]:return
 free,total=torch.cuda.mem_get_info();required=8_200_000_000+4*(16384+199)*36*2*8*128*2+4*1024**3
 if free<required:raise RuntimeError(f'Insufficient safe VRAM: {free} < {required}')
 torch.cuda.set_per_process_memory_fraction(.90);t=time.monotonic();model,tok=load_real_qwen(config(root));backend=BatchedQwen(config(root).server.thinking,model,tok)
 from .real_run import sha
 assert sha(tok.chat_template)==policy['observation_provenance']['template_hash'] and sha(tok.backend_tokenizer.to_str())==policy['observation_provenance']['tokenizer_hash']
 atomic_write_json(root/'collection'/f'load{index}.json',{'cold_seconds':time.monotonic()-t,'dtype':str(next(model.parameters()).dtype),'gpu':index,'scheduler':'homogeneous_prompt_groups','max_batch':4,'calibration_batch':2,'use_cache':True,'hooks':sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules()),'memory_fraction':.90,'free_vram_before_load':free})
 for stage,g in queue[index::4]:
  for p in g['prompts']:
   if guard.should_stop()[0]:break
   # One question, independent seeds: no left-padding mask or repeated KV expansion.
   batch_group={**g,'prompts':[p],'batch_id':g['original_id']+':'+p['prompt_id']}
   collect_group(root,stage,batch_group,backend,policy,guard);torch.cuda.empty_cache()
  print(json.dumps({'stage':stage,'original':g['original_id'],'gpu':index,'active_minutes':guard.elapsed()/60}),flush=True)
  if guard.should_stop()[0]:break
 del model;torch.cuda.empty_cache()

def main(root):
 commands=[(i,[sys.executable,'-m','aimo.bf16_bucket','worker',str(root),str(i)]) for i in range(4)]
 codes=run(root,commands,'collection')
 shared=json.loads((root/'collection/exit.json').read_text());shared['includes_concurrent_main_and_calibration']=True
 atomic_write_json(root/'calibration/exit.json',shared)
 from .bf16_report import summarize
 cal=summarize(root,'calibration');main_summary=summarize(root,'collection')
 for stage,d in [('calibration',cal),('collection',main_summary)]:
  d['wall_time_scope']='shared calibration+main worker stage, not additive across stages';atomic_write_json(root/stage/'summary.json',d)
 if any(c!=0 for c in codes):
  atomic_write_json(root/'runtime/blocked.json',{'reason':'worker infrastructure failure','codes':codes});return
 from .bf16_post import post
 post(root)
if __name__=='__main__':
 mode=sys.argv[1];root=Path(sys.argv[2])
 if mode=='worker':worker(root,int(sys.argv[3]))
 else:main(root)
